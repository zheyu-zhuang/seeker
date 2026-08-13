"""Core Seeker model: query composition, intent refinement, and stage outputs."""

import logging
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect
from einops import rearrange

from seeker.model.common.normalizer import MultiRobotLinearNormalizer
from seeker.model.dinov3_core.make_dinov3_vits import dinov3_vits16plus
from seeker.util.roi import get_square_box, grid_box_to_mask, norm_to_prob, nucleus
from seeker.util.seeker_config import (
    IntentRefinerConfig,
    QueryComposerConfig,
    SeekerModelConfig,
    build_seeker_pretty_config,
    load_seeker_base_config,
    resolve_seeker_model_config,
)
from seeker.util.formatting import pretty_print_nested
from seeker.util.task_meta import NUM_ROBOTS


@dataclass
class SeekerStageOut:
    """Outputs from one Seeker stage."""

    ctx: torch.Tensor
    attn_map: torch.Tensor
    head_score: torch.Tensor
    mask: torch.Tensor


@dataclass
class SeekerOut:
    """Seeker outputs grouped by stage."""

    coarse: SeekerStageOut
    fine: Optional[SeekerStageOut] = None

    @property
    def final(self) -> SeekerStageOut:
        """Return fine stage when available, otherwise coarse stage."""
        return self.fine if self.fine is not None else self.coarse


def _init_linear(m: nn.Linear, std: float = 0.02) -> None:
    nn.init.normal_(m.weight, std=std)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _init_layernorm(m: nn.LayerNorm) -> None:
    nn.init.ones_(m.weight)
    nn.init.zeros_(m.bias)


class QueryComposer(nn.Module):
    """Compose query tokens from proprioception, robot id, and task embedding."""

    def __init__(self, cfg: QueryComposerConfig) -> None:
        super().__init__()

        emb_dim = int(cfg.emb_dim)
        num_robots = int(cfg.num_robots)
        task_emb_dim = int(cfg.task_emb_dim)
        use_rotation = bool(cfg.use_rotation)
        hidden_mult = int(cfg.hidden_mult)
        proprio_dim = int(cfg.proprio_dim)
        disable_proprio = bool(cfg.disable_proprio)

        if proprio_dim <= 0:
            raise ValueError("proprio_dim must be > 0")

        self.num_robots = num_robots
        self.use_rotation = use_rotation
        self.disable_proprio = disable_proprio

        self.task_proj = nn.Sequential(
            nn.Linear(task_emb_dim, hidden_mult * emb_dim),
            nn.GELU(),
            nn.Linear(hidden_mult * emb_dim, emb_dim),
        )

        self.proprio_film = None
        if not disable_proprio:
            in_prop_dim = proprio_dim + num_robots + (6 if use_rotation else 0)
            self.proprio_film = nn.Sequential(
                nn.Linear(in_prop_dim, hidden_mult * emb_dim),
                nn.GELU(),
                nn.Linear(hidden_mult * emb_dim, 2 * emb_dim),
                nn.Tanh(),
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m)
            elif isinstance(m, nn.LayerNorm):
                _init_layernorm(m)

    def forward(self, composer_in: dict, noise: float = 0.0) -> torch.Tensor:
        task_token = self.task_proj(composer_in["task_embedding"])

        if self.disable_proprio:
            # No proprio/robot-id conditioning: query is a straight learnable
            # projection of the task token.
            return task_token

        eef_pos = composer_in["eef_pos"]
        gripper_opening = composer_in["gripper_opening"]
        robot_id = composer_in["robot_id"]

        prop = torch.cat([eef_pos, gripper_opening, robot_id], dim=-1)
        if self.use_rotation:
            prop = torch.cat([prop, composer_in["eef_rot"]], dim=-1)

        if noise > 0:
            prop = prop + torch.randn_like(prop) * noise

        gamma, beta = self.proprio_film(prop).chunk(2, dim=-1)
        return (1.0 + gamma) * task_token + beta


class IntentRefiner(nn.Module):
    """Iterative cross-attention refinement with fixed head gating."""

    def __init__(self, cfg: IntentRefinerConfig) -> None:
        super().__init__()

        emb_dim = int(cfg.emb_dim)
        num_heads = int(cfg.num_heads)
        n_iters = int(cfg.num_refinement_iters)
        entmax_alpha = float(cfg.entmax_alpha)
        hidden_multiplier = int(cfg.hidden_multiplier)
        disable_head_gating = bool(cfg.disable_head_gating)

        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")
        if n_iters < 1:
            raise ValueError("n_iters must be >= 1")
        if entmax_alpha <= 1.0:
            raise ValueError("entmax_alpha must be > 1.0 for sparsity")

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.n_iters = n_iters
        self.entmax_alpha = entmax_alpha
        self.disable_head_gating = disable_head_gating

        self.norm_q = nn.LayerNorm(emb_dim)
        self.norm_dh = nn.LayerNorm(self.head_dim)
        self.query_proj = nn.Linear(emb_dim, emb_dim)

        self.intent_proj = None
        if not disable_head_gating:
            self.intent_proj = nn.Linear(self.head_dim, self.head_dim)

        self.film = nn.Sequential(
            nn.Linear(self.head_dim, hidden_multiplier * self.head_dim),
            nn.GELU(),
            nn.Linear(hidden_multiplier * self.head_dim, 2 * emb_dim),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m)
            elif isinstance(m, nn.LayerNorm):
                _init_layernorm(m)

    def _step(
        self,
        *,
        q_state: torch.Tensor,
        v_h: torch.Tensor,
        k_t: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        update_query: bool = True,
    ):
        b, nq, _ = q_state.shape
        h, dh = self.num_heads, self.head_dim

        q_attn = self.query_proj(self.norm_q(q_state))
        q_attn = q_attn.view(b, nq, h, dh).permute(0, 2, 1, 3)

        scores = torch.matmul(q_attn, k_t)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn = entmax_bisect(scores, alpha=self.entmax_alpha, dim=-1)

        ctx_h = torch.matmul(attn, v_h)
        if self.disable_head_gating:
            # No learned head weighting: plain uniform average across heads.
            head_score = torch.full_like(ctx_h[..., :1], 1.0 / h)
        else:
            q_bar = q_attn.mean(dim=1)
            q_n = F.normalize(self.intent_proj(q_bar), dim=-1, eps=1e-6)
            ctx_n = F.normalize(ctx_h, dim=-1, eps=1e-6)
            logits = (ctx_n * q_n.unsqueeze(1)).sum(dim=-1, keepdim=True)
            head_score = F.softmax(logits, dim=1)
        ctx = (ctx_h * head_score).sum(dim=1)

        q_next = q_state
        if update_query:
            gamma, beta = self.film(self.norm_dh(ctx)).chunk(2, dim=-1)
            q_next = (1.0 + gamma) * q_state + beta

        return ctx, attn, head_score, q_next

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_all_iters: bool = False,
    ):
        """Refine query against key/value tokens and return final or per-iteration outputs."""
        b, nk, c = k.shape
        h, dh = self.num_heads, self.head_dim

        if v.shape != k.shape or c != self.emb_dim:
            raise ValueError("k/v shapes inconsistent with emb_dim")

        if q.dim() == 2:
            q_state = q[:, None, :]
        elif q.dim() == 3:
            q_state = q
        else:
            raise ValueError(f"q must be [B,C] or [B,Nq,C], got {tuple(q.shape)}")

        if q_state.size(-1) != c:
            raise ValueError("k/v/q shapes are inconsistent")

        k_h = k.reshape(b, nk, h, dh).permute(0, 2, 1, 3)
        v_h = v.reshape(b, nk, h, dh).permute(0, 2, 1, 3)
        k_t = k_h.transpose(-2, -1)

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise TypeError("attn_mask must be boolean")
            if attn_mask.dim() == 3:
                attn_mask = attn_mask[:, None]

        iter_outs = []
        for i in range(self.n_iters):
            ctx, attn, head_score, q_state = self._step(
                q_state=q_state,
                v_h=v_h,
                k_t=k_t,
                attn_mask=attn_mask,
                update_query=(i < self.n_iters - 1),
            )
            if return_all_iters:
                iter_outs.append([ctx, attn, head_score])

        if return_all_iters:
            return iter_outs
        return ctx, attn, head_score


class SeekerStage(nn.Module):
    """Single Seeker stage for one camera view."""

    def __init__(self, cfg: SeekerModelConfig) -> None:
        super().__init__()

        query_cfg = cfg.query_composer
        refiner_cfg = cfg.intent_refiner
        emb_dim = int(query_cfg.emb_dim)
        self.query_composer = QueryComposer(query_cfg)
        self.refiner = IntentRefiner(refiner_cfg)

        self.to_k = nn.Linear(emb_dim, emb_dim)
        self.to_v = nn.Linear(emb_dim, emb_dim)
        _init_linear(self.to_k)
        _init_linear(self.to_v)

    def forward(
        self,
        x: torch.Tensor,
        composer_in: dict,
        proprio_noise: float = 0.0,
        attn_mask: Optional[torch.Tensor] = None,
        return_all_iters: bool = False,
    ):
        return self.refiner(
            k=self.to_k(x),
            v=self.to_v(x),
            q=self.query_composer(composer_in, noise=proprio_noise),
            attn_mask=attn_mask,
            return_all_iters=return_all_iters,
        )


class Seeker(nn.Module):
    """Seeker with frozen DINOv3 backbone and per-view coarse/fine branches."""

    def __init__(
        self,
        config: Optional[dict] = None,
        num_robots: int = NUM_ROBOTS,
        verbose: bool = True,
        views: Optional[List[str]] = None,
        weights: str = "",
        strict_weights: bool = False,
    ) -> None:
        super().__init__()

        self.views = views or ["agentview"]
        base_cfg = load_seeker_base_config(overrides=config)

        self.vit = self._init_dinov3(
            ckpt_path=base_cfg.backbone.ckpt_path,
        )
        self.num_heads = self.vit.num_heads
        self.emb_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_size

        cfg = resolve_seeker_model_config(
            base_cfg,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            num_robots=int(num_robots),
        )

        self.out_dim = self.emb_dim // self.num_heads
        self.top_p = float(cfg.seeker.top_p)
        self.select_n_heads = int(cfg.seeker.select_n_heads)
        self.disable_head_gating = bool(cfg.intent_refiner.disable_head_gating)
        self.backbone_name = str(cfg.backbone.name)
        self.task_emb_dim = int(cfg.query_composer.task_emb_dim)

        self.model_cfg = cfg

        self.view_branches = nn.ModuleDict(
            {
                view: nn.ModuleDict(
                    {
                        "coarse": SeekerStage(cfg),
                        "fine": SeekerStage(cfg),
                    }
                )
                for view in self.views
            }
        )

        self.normalizer = MultiRobotLinearNormalizer()
        self.ckpt_path = None
        self.load_pretrained_weights(weights=weights, strict=strict_weights)

        if verbose:
            pretty_print_nested(self.get_config(), title="Seeker Configuration")

    @staticmethod
    def _init_dinov3(ckpt_path: str) -> nn.Module:
        logging.getLogger("dinov3").setLevel(logging.WARNING)
        vit = dinov3_vits16plus(pretrained=False)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                "DINOv3 checkpoint not found: "
                f"{ckpt_path}. "
                "Please follow "
                "'https://github.com/facebookresearch/dinov3' "
                "to download the pretrained weights and provide the correct path "
                "in the config."
            )
        vit.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
        for p in vit.parameters():
            p.requires_grad = False
        return vit

    def _extract_patch_features(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vit.forward_features(image)["x_norm_patchtokens"]

    def set_normalizer(self, normalizer: MultiRobotLinearNormalizer) -> None:
        """Copy normalizer state used by observation preprocessors."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(
        self,
        image: torch.Tensor,
        view: str,
        composer_in: dict,
        stage: str = "coarse",
        *,
        proprio_noise: float = 0.0,
    ) -> SeekerOut:
        """Run Seeker on one view and return stage-structured outputs."""
        if stage not in ("coarse", "fine"):
            raise ValueError(f"Unknown stage: {stage}")

        x = self._extract_patch_features(image)
        shared_kwargs = {
            "x": x,
            "composer_in": composer_in,
            "proprio_noise": proprio_noise,
        }

        # Coarse stage always runs.
        ctx, attn_map, head_score = self.view_branches[view]["coarse"](**shared_kwargs)
        mask = self.mask_from_attention(attn_map, head_score)
        coarse_out = SeekerStageOut(
            ctx=ctx,
            attn_map=attn_map,
            head_score=head_score,
            mask=mask,
        )
        fine_out = None

        if stage == "fine":
            # Fine stage is restricted to the coarse mask region.
            box = get_square_box(mask.float().mean(dim=1))
            shared_kwargs["attn_mask"] = grid_box_to_mask(box, S=mask.size(-1))
            ctx, attn_map, head_score = self.view_branches[view]["fine"](
                **shared_kwargs
            )
            mask = self.mask_from_attention(attn_map, head_score)
            fine_out = SeekerStageOut(
                ctx=ctx,
                attn_map=attn_map,
                head_score=head_score,
                mask=mask,
            )

        return SeekerOut(
            coarse=coarse_out,
            fine=fine_out,
        )

    def mask_from_attention(
        self,
        attn_map: torch.Tensor,
        head_score: torch.Tensor,
    ) -> torch.Tensor:
        """Build spatial masks from attention maps and head scores."""
        if attn_map.dim() != 4:
            raise ValueError("attn_map must be [B, H, Nq, HW]")
        if head_score is None or head_score.dim() != 4:
            raise ValueError("head_score must be [B, H, Nq, 1]")

        b, h, nq, hw = attn_map.shape
        s = math.isqrt(hw)
        if s * s != hw:
            raise ValueError("HW must be a perfect square")
        if not (0 < self.select_n_heads <= h):
            raise ValueError("select_n_heads must be in [1, num_heads]")

        attn = norm_to_prob(rearrange(attn_map, "b h nq hw -> (b nq) h hw"))
        sel_scores = rearrange(head_score.squeeze(-1), "b h nq -> (b nq) h")

        bxnq = b * nq
        k = h if self.disable_head_gating else self.select_n_heads
        if k < h:
            sel_idx = torch.topk(sel_scores, k=k, dim=1, largest=True).indices
            w = torch.gather(sel_scores, 1, sel_idx)
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        else:
            sel_idx = torch.arange(h, device=attn.device).expand(bxnq, h)
            w = torch.full((bxnq, h), 1.0 / h, device=attn.device, dtype=attn.dtype)

        l = attn.size(-1)
        attn_sel = torch.gather(attn, 1, sel_idx.unsqueeze(-1).expand(-1, -1, l))
        mix = norm_to_prob((attn_sel * w.unsqueeze(-1)).sum(dim=1))

        mask_flat, _ = nucleus(mix, top_p=self.top_p, min_tokens=1)
        mask_flat = norm_to_prob(mask_flat)

        mask = rearrange(mask_flat, "(b nq) hw -> b nq hw", b=b, nq=nq, hw=hw)
        return rearrange(mask, "b nq (h w) -> b nq h w", h=s, w=s)

    def load_pretrained_weights(
        self, weights: Optional[str] = None, *, strict: bool = False
    ) -> None:
        """Load Seeker checkpoint weights when provided."""
        if not weights:
            self.ckpt_path = None
            return

        if not os.path.exists(weights):
            raise FileNotFoundError(f"Seeker checkpoint not found: {weights}")

        state_dict = torch.load(weights, map_location="cpu")
        state_dict = self._normalize_checkpoint_state_dict(state_dict)
        incompat = self.load_state_dict(state_dict, strict=False)

        if strict:
            missing_core = [
                key for key in incompat.missing_keys if not str(key).startswith("vit.")
            ]
            unexpected_core = [
                key for key in incompat.unexpected_keys
                if not str(key).startswith("vit.")
            ]
            if missing_core or unexpected_core:
                msg = []
                if missing_core:
                    preview = ", ".join(missing_core[:10])
                    suffix = " ..." if len(missing_core) > 10 else ""
                    msg.append(
                        f"{len(missing_core)} missing core keys: {preview}{suffix}"
                    )
                if unexpected_core:
                    preview = ", ".join(unexpected_core[:10])
                    suffix = " ..." if len(unexpected_core) > 10 else ""
                    msg.append(
                        f"{len(unexpected_core)} unexpected core keys: {preview}{suffix}"
                    )
                raise RuntimeError(
                    "Seeker checkpoint is incompatible with v1.0 core: "
                    + "; ".join(msg)
                )
            self.ckpt_path = weights
            return

        if incompat.missing_keys:
            preview = ", ".join(incompat.missing_keys[:10])
            suffix = " ..." if len(incompat.missing_keys) > 10 else ""
            warnings.warn(
                f"Seeker partial checkpoint load: {len(incompat.missing_keys)} missing keys: "
                f"{preview}{suffix}",
                stacklevel=2,
            )
        if incompat.unexpected_keys:
            preview = ", ".join(incompat.unexpected_keys[:10])
            suffix = " ..." if len(incompat.unexpected_keys) > 10 else ""
            warnings.warn(
                f"Seeker partial checkpoint load: {len(incompat.unexpected_keys)} unexpected keys: "
                f"{preview}{suffix}",
                stacklevel=2,
            )

        self.ckpt_path = weights

    @staticmethod
    def _normalize_checkpoint_state_dict(state_dict):
        """Accept standalone Seeker weights and minimal wrapper-prefixed variants."""
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                value = state_dict.get(key)
                if isinstance(value, dict):
                    state_dict = value
                    break

        if not isinstance(state_dict, dict):
            return state_dict

        prefixes = (
            "module.",
            "seeker.",
            "obs_encoder.seeker.",
            "policy.obs_encoder.seeker.",
        )
        keys = list(state_dict.keys())
        for prefix in prefixes:
            if keys and all(str(key).startswith(prefix) for key in keys):
                return {
                    str(key)[len(prefix) :]: value
                    for key, value in state_dict.items()
                }
        return state_dict

    def get_config(self) -> dict:
        """Return a readable config snapshot for logging/debugging."""
        return build_seeker_pretty_config(
            view_names=self.view_branches.keys(),
            ckpt_path=self.ckpt_path,
            cfg=self.model_cfg,
            out_dim=self.out_dim,
        )
