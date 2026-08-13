"""Training encoder that wraps Seeker for policy learning."""

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from seeker.model.seeker_model import Seeker, SeekerStageOut
from seeker.model.base_encoder import BaseEncoder, EncoderInputs
from seeker.util.visual_randomizer import CropRandomizer, BackgroundRandomizer
from seeker.util.roi import grid_mask_to_pixel_box, box_px_to_grid_mask
from seeker.util.image_ops import resize_image
from seeker.util.visualization import visualize


@dataclass
class EncoderStageOut:
    """Encoder wrapper around one Seeker stage output."""

    stage: str
    image: torch.Tensor
    seeker: SeekerStageOut
    tight_box: torch.Tensor  # grid_mask_to_pixel_box output (used for norm + loss)
    input_box: Optional[torch.Tensor]  # box used to create this stage input
    feat: torch.Tensor

    @property
    def ctx(self) -> torch.Tensor:
        return self.seeker.ctx

    @property
    def attn(self) -> torch.Tensor:
        return self.seeker.attn_map

    @property
    def head_score(self) -> torch.Tensor:
        return self.seeker.head_score

    @property
    def mask(self) -> torch.Tensor:
        return self.seeker.mask


class SeekerTrainingEncoder(BaseEncoder):
    """Seeker-backed visual encoder used during training.

    Supports three modes via `visual_mode`:
    - `agentview`: external camera only
    - `agentview_eih`: external + eye-in-hand
    - `finetune_eih`: train eye-in-hand with cached agentview coarse features
    """

    def __init__(
        self,
        image_size: int,
        input_res: int = 224,
        n_hidden: int = 128,
        obs_dropout: float = 0.0,
        *,
        visual_mode: str = "agentview",  # | "agentview_eih" | "finetune_eih"
        background_path: Optional[str] = None,
        weights: Optional[str] = None,
        seeker_config: Optional[dict] = None,
    ) -> None:
        """Initialize SeekerTrainingEncoder.

        Args:
            image_size: Input image resolution before crop/resize.
            input_res: Resolution passed into Seeker.
            n_hidden: Hidden dimension for projection layers
            obs_dropout: Dropout probability for observation features
            visual_mode: One of "agentview", "agentview_eih", "finetune_eih".
            background_path: Optional texture/background directory for overlay augmentation.
            weights: Optional Seeker checkpoint path (used in finetune_eih).
            seeker_config: Optional overrides merged onto seeker_default_params.yaml
                (e.g. intent_refiner.num_refinement_iters, query_composer.disable_proprio).
        """
        super().__init__()
        assert visual_mode in (
            "agentview",
            "agentview_eih",
            "finetune_eih",
        ), f"Unknown visual_mode: {visual_mode}"

        # Core modules
        obs_shape = (3, image_size, image_size)
        self.crop_randomizer = CropRandomizer(obs_shape, crop_size=input_res)

        self.visual_mode = visual_mode

        self.enable_eih = visual_mode != "agentview"
        self.use_cached_agentview = visual_mode == "finetune_eih"

        views = ["agentview"]
        if self.enable_eih:
            views.append("eye_in_hand")

        weights = weights if self.visual_mode == "finetune_eih" else None

        self.seeker = Seeker(
            views=views, weights=weights, verbose=False, config=seeker_config
        )

        # Feature projections
        feat_dim = self.seeker.out_dim
        self.task_emb_proj = nn.Linear(self.seeker.task_emb_dim, 16)
        in_dim = feat_dim + 11 + 16 + self.num_robots  # proprio + task_emb + robot_id
        in_dim += feat_dim if self.enable_eih else 0

        self.fine_proj = nn.Linear(in_dim, n_hidden)
        self.coarse_proj = nn.Linear(in_dim, n_hidden)

        # Training parameters
        self.obs_dropout = obs_dropout
        self.proprio_noise = 0.005

        # Processing parameters
        self.input_res = input_res
        self.image_size = image_size
        self.margin = 8
        self.box_jitter = 0.1

        # Visualization and caching
        self.visualize_frequency = 1000
        self.counter = 0
        self.buffer = None
        self.viz_dir = None
        self.disable_random_crop = False

        # background randomizer
        if background_path is not None:
            self.background_randomizer = BackgroundRandomizer(
                input_shape=(self.input_res, self.input_res),
                background_path=background_path,
            )
        else:
            self.background_randomizer = None

    def set_normalizer(self, normalizer):
        self.seeker.set_normalizer(normalizer)

    def initialize_internal(self, dataset_size, device, viz_dir):
        self.viz_dir = viz_dir
        if not self.use_cached_agentview:
            return

        feat_dim = self.seeker.out_dim
        self.buffer = torch.empty(
            dataset_size, feat_dim, dtype=torch.float32, device=device
        )
        self.buffer_valid = torch.zeros(dataset_size, dtype=torch.bool, device=device)

    def forward(
        self,
        obs,
        *,
        stage: str = "coarse",
        overlay_alpha=None,
        task_instruction=None,
        obs_index=None,
    ):
        shared_args = dict(
            obs=obs,
            overlay_alpha=overlay_alpha,
            task_instruction=task_instruction,
            stage=stage,
        )
        if self.use_cached_agentview:
            return self._forward_finetune_eih(obs_index=obs_index, **shared_args)
        else:
            return self._forward(**shared_args)

    def _forward_finetune_eih(
        self,
        obs,
        *,
        stage: str,
        obs_index,
        overlay_alpha=None,
        task_instruction=None,
    ):
        """Forward path for `finetune_eih` mode using cached agentview coarse features."""
        assert self.use_cached_agentview
        assert self.enable_eih
        assert self.buffer is not None, "Call initialize_internal() before finetuning."
        assert obs_index is not None, "obs_index required for caching."
        if obs_index.shape[-1] == 1:
            obs_index = obs_index.squeeze(-1)
        assert obs_index.dim() == 1, "obs_index must be shape [B]."

        # Keep deterministic agentview preprocessing so cache lookup is stable.
        enc_in = self.process_obs(obs)
        assert enc_in.T == 1, (
            "finetune_eih assumes T==1 (frame-level indexing). "
            "If T>1, pass per-frame indices or reshape indices accordingly."
        )

        if self.enable_eih:
            assert enc_in.eye_in_hand is not None, "eye-in-hand image is missing"

        # 1) Retrieve cached agentview coarse ctx (or populate cache).
        idx = obs_index.to(self.buffer.device, non_blocking=True)
        need_cache = ~self.buffer_valid[idx]  # [B] bool

        if need_cache.any():
            # Compute agentview coarse ctx only for missing entries.
            with torch.no_grad():
                agentview = enc_in.agentview[need_cache]
                composer_in = {k: v[need_cache] for k, v in enc_in.composer_in.items()}
                agent_outs_missing = self.run_stages(
                    agentview,
                    stage="coarse",
                    view="agentview",
                    composer_in=composer_in,
                    overlay_alpha=None,
                )
                assert (
                    len(agent_outs_missing) > 0
                ), "No stage output while populating cache."
                agent_ctx_missing = (
                    agent_outs_missing[0].feat.detach().float()
                )  # [Bm, D]

            self.buffer[idx[need_cache]] = agent_ctx_missing
            self.buffer_valid[idx[need_cache]] = True

        agent_ctx = self.buffer[idx]  # [B, D]

        # 2) Compute EIH coarse ctx (trainable path).
        eih_outs = self.run_stages(
            enc_in.eye_in_hand,
            stage=stage,
            view="eye_in_hand",
            composer_in=enc_in.composer_in,
            overlay_alpha=overlay_alpha,
        )
        assert len(eih_outs) > 0, "EIH stages empty."

        self.save_images(eih_outs)
        self.counter += 1

        # Randomly drop agentview features during training.
        mask_prob = 0.2
        if self.training and mask_prob > 0.0:
            mask = torch.rand(agent_ctx.size(0), 1, device=agent_ctx.device) > mask_prob
            agent_ctx = agent_ctx * mask.float()

        agent_feats = [agent_ctx for _ in range(len(eih_outs))]

        feat_dict = self.aggregate_features(
            enc_in=enc_in,
            agent_feats=agent_feats,
            eih_feats=[out.feat for out in eih_outs],
        )

        return feat_dict, self.stage_consistency_loss(eih_outs)

    def _forward(
        self,
        obs,
        *,
        stage: str,
        overlay_alpha=None,
        task_instruction=None,
    ):
        assert stage in ["fine", "coarse"], f"Unknown Stage: {stage}"

        enc_in = self.process_obs(obs)

        shared = dict(composer_in=enc_in.composer_in, overlay_alpha=overlay_alpha)

        agent_outs = self.run_stages(
            enc_in.agentview, stage=stage, view="agentview", **shared
        )
        eih_outs = self.run_stages(
            enc_in.eye_in_hand, stage=stage, view="eye_in_hand", **shared
        )

        if self.enable_eih:
            assert len(eih_outs) > 0, "eye-in-hand image is missing"

        self.save_images(agent_outs, eih_outs=eih_outs)
        self.counter += 1

        eih_feats = [out.feat for out in eih_outs] if self.enable_eih else None
        feat_dict = self.aggregate_features(
            enc_in=enc_in,
            agent_feats=[out.feat for out in agent_outs],
            eih_feats=eih_feats,
        )

        consistency_loss = self.stage_consistency_loss(agent_outs)
        consistency_loss += self.stage_consistency_loss(eih_outs)

        return feat_dict, consistency_loss

    def run_stages(
        self,
        image: torch.Tensor,
        view: str,
        *,
        composer_in,
        stage,
        overlay_alpha=None,
    ) -> list[EncoderStageOut]:
        """Run Seeker stages on one view and wrap outputs for encoder use."""

        if image is None:
            return []

        assert stage in ("coarse", "fine")
        assert image.shape[-1] in (self.input_res, self.input_res // 2)

        image_aug = self._random_overlay(image, overlay_alpha)

        B, _, H, W = image_aug.shape
        device = image_aug.device
        full_box = torch.tensor([[0, 0, W - 1, H - 1]], device=device).expand(B, -1)

        out = self.seeker(
            image=image_aug,
            view=view,
            composer_in=composer_in,
            proprio_noise=self.proprio_noise,
            stage=stage,
        )

        stage_outs = [("coarse", out.coarse)]
        if out.fine is not None:
            stage_outs.append(("fine", out.fine))

        outs: list[EncoderStageOut] = []
        for stage_name, stage_out in stage_outs:
            ctx = stage_out.ctx
            mask = stage_out.mask
            tight_box = grid_mask_to_pixel_box(mask.squeeze(1), full_box)

            # Store raw ctx as (B, D), assuming Nq == 1.
            ctx_vec = ctx.squeeze(1)

            outs.append(
                EncoderStageOut(
                    stage=stage_name,
                    image=image_aug,
                    seeker=stage_out,
                    tight_box=tight_box,
                    input_box=full_box,
                    feat=ctx_vec,
                )
            )
        return outs

    def aggregate_features(
        self,
        *,
        agent_feats: list[torch.Tensor],
        eih_feats: Optional[list[torch.Tensor]],
        enc_in,
    ) -> dict:
        assert len(agent_feats) > 0, "agent_feats cannot be empty."

        if self.enable_eih:
            assert eih_feats is not None and len(eih_feats) == len(
                agent_feats
            ), "Stage mismatch"

        proprio = enc_in.proprio
        task_embedding = enc_in.task_embedding
        robot_id = enc_in.composer_in["robot_id"]

        task_ind = self.task_emb_proj(task_embedding)
        aux = torch.cat([proprio, task_ind, robot_id], dim=-1)

        # -------- stage 0 (coarse) --------
        coarse_in = torch.cat([agent_feats[0], aux], dim=-1)
        if self.enable_eih:
            coarse_in = torch.cat([eih_feats[0], coarse_in], dim=-1)
        coarse_feat = self.coarse_proj(coarse_in)

        # -------- stage 1 (fine, optional) --------
        fine_feat = None
        if len(agent_feats) > 1:
            fine_in = torch.cat([agent_feats[1], aux], dim=-1)
            if self.enable_eih:
                fine_in = torch.cat([eih_feats[1], fine_in], dim=-1)
            fine_feat = self.fine_proj(fine_in)

        feat_dict = {"coarse": coarse_feat, "fine": fine_feat}

        # global feature dropout
        if self.obs_dropout > 0.0 and self.training:
            for k, v in feat_dict.items():
                if v is not None:
                    feat_dict[k] = F.dropout(v, p=self.obs_dropout, training=True)

        return feat_dict

    def save_images(
        self,
        outs: list[EncoderStageOut],
        *,
        eih_outs: Optional[list[EncoderStageOut]] = None,
        text: str = None,
    ):
        if (not self.training) or (self.counter % self.visualize_frequency != 0):
            return

        if len(outs) == 0:
            return

        # Agentview rows
        coarse = outs[0]
        fine = outs[1] if len(outs) > 1 else None
        eih = eih_outs[0] if eih_outs is not None and len(eih_outs) > 0 else None

        coarse_box = coarse.tight_box
        fine_box = fine.tight_box if fine is not None else None

        coarse_plain = [
            coarse.image,
            coarse.mask,
            None,
            None,
        ]

        coarse_view = [
            coarse.image,
            coarse.mask,
            [coarse_box] if fine is not None else coarse_box,
            None,
        ]

        views = [coarse_plain, coarse_view]

        if fine is not None:
            fine_view = [fine.image, fine.mask, fine_box, None]
            views.append(fine_view)

        # Eye-in-hand row (optional)
        if eih is not None:
            eih = eih_outs[0]  # coarse-only by design
            eih_view = [eih.image, eih.mask, eih.tight_box, None]
            views.append(eih_view)

        visualize(views, save_dir=self.viz_dir, step=self.counter, text=text)

    def stage_consistency_loss(
        self, stages: list[EncoderStageOut], pad_box=True
    ) -> torch.Tensor:
        """KL consistency between coarse attention and final-stage tight box."""
        if stages is None or len(stages) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        coarse_attn = stages[0].attn  # [(B*T), H, Nq, Nk] or [B, H, Nq, Nk]
        tight_box = stages[-1].tight_box  # [B,4] or [(B*T),4]
        # pad tight_box to avoid cutting off attention at borders
        if pad_box:
            tight_box = tight_box.clone()
            tight_box[:, 0] = torch.clamp(tight_box[:, 0] - self.margin, min=0)
            tight_box[:, 1] = torch.clamp(tight_box[:, 1] - self.margin, min=0)
            tight_box[:, 2] = torch.clamp(
                tight_box[:, 2] + self.margin, max=self.input_res - 1
            )
            tight_box[:, 3] = torch.clamp(
                tight_box[:, 3] + self.margin, max=self.input_res - 1
            )
        head_score = stages[0].head_score  # [B, H, Nq, 1] or [B, H, Nq]

        # Select top-k heads per (b, nq)
        head_score = rearrange(head_score.squeeze(-1), "b h nq -> (b nq) h")
        k = int(self.seeker.select_n_heads)
        sel_heads = torch.topk(head_score, k=k, dim=1, largest=True).indices

        # Gather selected heads
        sel_idx = (
            sel_heads.unsqueeze(-1).expand(-1, -1, coarse_attn.size(-1)).unsqueeze(2)
        )
        coarse_attn_sel = torch.gather(coarse_attn, 1, sel_idx)

        # Trim attention with tight box
        Nk = coarse_attn_sel.shape[-1]
        S = int(Nk**0.5)
        assert Nk == S * S, f"Expected square grid, got {Nk} != {S}^2"

        box_grid = box_px_to_grid_mask(tight_box, img_size=self.input_res, S=S)
        trimmed = coarse_attn_sel * box_grid.unsqueeze(1)

        # Flatten (time may already be folded into batch)
        coarse_attn_sel = coarse_attn_sel.reshape(-1, Nk)
        trimmed = trimmed.reshape(-1, Nk)

        # KL divergence
        eps = 1e-8
        q = trimmed / (trimmed.sum(dim=-1, keepdim=True) + eps)  # teacher
        p = coarse_attn_sel / (
            coarse_attn_sel.sum(dim=-1, keepdim=True) + eps
        )  # student
        p = p.clamp(min=1e-4)

        return F.kl_div(p.log(), q.detach(), reduction="none").mean()

    def _random_overlay(self, image, overlay_alpha):
        if self.background_randomizer is None or overlay_alpha is None:
            return image
        B, C, H, W = image.shape
        bg = self.background_randomizer(B)
        if bg.shape[-2:] != (H, W):
            bg = F.interpolate(bg, size=(H, W), mode="bilinear", align_corners=False)
        a = overlay_alpha if overlay_alpha is not None else 0.5
        rand_indices = torch.randperm(B)[: int(B * 0.5)]
        img_aug = image.clone()
        img_aug[rand_indices] = image[rand_indices] * a + bg[rand_indices] * (1 - a)
        return img_aug

    def process_obs(self, obs: Dict[str, torch.Tensor]) -> EncoderInputs:
        """Normalize and preprocess observations for Seeker forward."""
        enc_in = self.obs_to_input(obs, self.seeker.normalizer, resize=False)

        def _proc_img(image, resize_only: bool = False):
            if image is None:
                return None
            if resize_only or self.disable_random_crop:
                return resize_image(image, self.input_res)
            return self.crop_randomizer(image)

        # Disable random crop on cached agentview path for consistency.
        enc_in.agentview = _proc_img(
            enc_in.agentview, resize_only=self.use_cached_agentview
        )
        enc_in.eye_in_hand = _proc_img(enc_in.eye_in_hand)
        return enc_in
