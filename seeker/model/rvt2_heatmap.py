"""RVT2Heatmap visual-focus model."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from seeker.model.common.dino import load_frozen_dinov3_vits16plus
from seeker.util.task_meta import NUM_ROBOTS
from seeker.focus_toolbox.prediction import VisualFocusPrediction
from seeker.focus_toolbox.config import load_rvt2_heatmap_config


class PatchFeatureBackbone(nn.Module):
    """Configurable patch-token backbone for heatmap localization."""

    def __init__(
        self,
        *,
        backbone_type: str,
        dino_ckpt_path: str | Path | None,
        image_size: int,
        patch_size: int,
        conv_dim: int = 384,
    ) -> None:
        super().__init__()
        self.backbone_type = str(backbone_type).strip().lower()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        if self.backbone_type == "dino":
            if dino_ckpt_path is None:
                raise ValueError("dino_ckpt_path is required for DINO backbone")
            self.encoder = load_frozen_dinov3_vits16plus(
                Path(dino_ckpt_path).expanduser(),
                device=torch.device("cpu"),
            )
            self.output_dim = int(getattr(self.encoder, "embed_dim"))
        elif self.backbone_type == "conv":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, conv_dim // 2, kernel_size=7, stride=2, padding=3),
                nn.GroupNorm(8, conv_dim // 2),
                nn.GELU(),
                nn.Conv2d(conv_dim // 2, conv_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, conv_dim),
                nn.GELU(),
                nn.Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=max(1, self.patch_size // 4),
                    stride=max(1, self.patch_size // 4),
                ),
            )
            self.output_dim = int(conv_dim)
        else:
            raise ValueError(
                f"Unknown patch backbone {backbone_type!r}; expected 'dino' or 'conv'"
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == "dino":
            return _extract_patch_features(self.encoder, image)
        x = self.encoder(image)
        grid_size = self.image_size // self.patch_size
        if x.shape[-2:] != (grid_size, grid_size):
            x = F.adaptive_avg_pool2d(x, output_size=(grid_size, grid_size))
        return x.flatten(2).transpose(1, 2)


class PatchActivationHead(nn.Module):
    """RVT-style transformer heatmap head over patch tokens."""

    def __init__(
        self,
        patch_dim: int,
        gripper_dim: int = 1,
        hidden_dim: int = 256,
        grid_size: int | None = None,
        task_emb_dim: int = 512,
        num_robots: int = NUM_ROBOTS,
        query_hidden_mult: int = 4,
        proprio_mode: str = "gripper_only",
        proprio_dim: int = 1,
        language_seq_len: int = 77,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        if int(gripper_dim) != 1:
            raise ValueError("PatchActivationHead expects scalar gripper opening")
        self.patch_dim = int(patch_dim)
        self.num_robots = int(num_robots)
        self.proprio_mode = str(proprio_mode)
        self.hidden_dim = int(hidden_dim)
        self.grid_size = None if grid_size is None else int(grid_size)
        self.language_seq_len = int(language_seq_len)
        self.num_patch_tokens = None if self.grid_size is None else self.grid_size**2
        _ = query_hidden_mult, proprio_dim
        self.patch_proj = nn.Linear(self.patch_dim, self.hidden_dim)
        self.lang_proj = nn.Linear(int(task_emb_dim), self.hidden_dim)
        self.gripper_token = nn.Linear(1, self.hidden_dim)
        self.robot_token = nn.Linear(self.num_robots, self.hidden_dim)
        self.patch_pos = (
            nn.Parameter(torch.empty(1, self.num_patch_tokens, self.hidden_dim))
            if self.num_patch_tokens is not None
            else None
        )
        self.lang_pos = nn.Parameter(
            torch.empty(1, self.language_seq_len, self.hidden_dim)
        )
        self.proprio_pos = nn.Parameter(torch.empty(1, 2, self.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(transformer_heads),
            dim_feedforward=4 * self.hidden_dim,
            dropout=float(transformer_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(transformer_depth),
            enable_nested_tensor=False,
        )
        self.heatmap_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        if self.patch_pos is not None:
            nn.init.normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.lang_pos, std=0.02)
        nn.init.normal_(self.proprio_pos, std=0.02)

    def _robot_one_hot(self, robot_id: torch.Tensor) -> torch.Tensor:
        if robot_id.ndim == 1 or (robot_id.ndim == 2 and robot_id.shape[-1] == 1):
            robot_id = robot_id.reshape(-1).long()
            return F.one_hot(robot_id, num_classes=self.num_robots).to(
                dtype=torch.float32
            )
        if robot_id.ndim == 2 and robot_id.shape[-1] == self.num_robots:
            return robot_id.to(dtype=torch.float32)
        raise ValueError(
            "robot_id must be [B], [B,1], or one-hot [B,num_robots], got "
            f"{tuple(robot_id.shape)}"
        )

    def forward(
        self,
        patches: torch.Tensor,
        gripper: torch.Tensor,
        task_embedding: torch.Tensor,
        robot_id: torch.Tensor,
        eef_pos: torch.Tensor | None = None,
        task_language_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = task_embedding, eef_pos
        if task_language_tokens is None:
            raise ValueError(
                "PatchActivationHead requires task_language_tokens; "
                "the pooled 512-D task_embedding fallback has been removed."
            )
        robot_one_hot = self._robot_one_hot(robot_id).to(
            device=patches.device,
            dtype=patches.dtype,
        )

        patch_tokens = self.patch_proj(patches)
        if self.patch_pos is not None:
            if patch_tokens.shape[1] != self.num_patch_tokens:
                raise ValueError(
                    f"Expected {self.num_patch_tokens} patch tokens, got "
                    f"{patch_tokens.shape[1]}"
                )
            patch_tokens = patch_tokens + self.patch_pos.to(
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )

        task_tokens = self.lang_proj(
            task_language_tokens.to(device=patches.device, dtype=patches.dtype)
        )
        if task_tokens.shape[1] != self.language_seq_len:
            raise ValueError(
                f"Expected {self.language_seq_len} language tokens, got "
                f"{task_tokens.shape[1]}"
            )
        task_tokens = task_tokens + self.lang_pos.to(
            device=task_tokens.device,
            dtype=task_tokens.dtype,
        )

        gripper_token = self.gripper_token(gripper).unsqueeze(1)
        robot_token = self.robot_token(robot_one_hot).unsqueeze(1)
        proprio_tokens = torch.cat([gripper_token, robot_token], dim=1)
        proprio_tokens = proprio_tokens + self.proprio_pos.to(
            device=proprio_tokens.device,
            dtype=proprio_tokens.dtype,
        )

        tokens = torch.cat([task_tokens, proprio_tokens, patch_tokens], dim=1)
        tokens = self.transformer(tokens)
        patch_tokens = tokens[:, task_tokens.shape[1] + proprio_tokens.shape[1] :]
        return self.heatmap_head(patch_tokens).squeeze(-1)


def _extract_patch_features(vit: nn.Module, image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return vit.forward_features(image)["x_norm_patchtokens"]


def load_checkpoint_payload(path: str | Path, *, map_location="cpu") -> dict:
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"RVT2Heatmap checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(payload, dict) or "head_state_dict" not in payload:
        raise ValueError(f"Invalid RVT2Heatmap checkpoint: {ckpt_path}")
    return payload


class RVT2Heatmap(nn.Module):
    """Load and run the RVT2Heatmap as a visual-focus model."""

    def __init__(
        self,
        *,
        checkpoint: str,
        vit_in: int,
        zoom: Optional[float] = None,
    ) -> None:
        super().__init__()
        if not checkpoint:
            raise ValueError(
                "rvt2_heatmap source requires "
                "policy.obs_encoder.focus.checkpoint"
            )

        self.checkpoint = str(checkpoint)
        self.vit_in = int(vit_in)
        self.config = load_rvt2_heatmap_config()
        self.patch_size = int(self.config["patch_size"])
        if self.vit_in % self.patch_size != 0:
            raise ValueError(
                f"vit_in={self.vit_in} must be divisible by RVT2Heatmap "
                f"patch_size={self.patch_size}"
            )
        self.grid_res = self.vit_in // self.patch_size
        self.zoom = (
            float(self.config["keypoint_box_zoom"])
            if zoom is None
            else float(zoom)
        )
        if self.zoom <= 0.0:
            raise ValueError(f"RVT2Heatmap zoom must be > 0, got {self.zoom}")

        payload = load_checkpoint_payload(self.checkpoint, map_location="cpu")
        if (
            "gripper_mean" not in payload
            or "gripper_std" not in payload
            or "head_config" not in payload
        ):
            raise ValueError(
                "RVT2Heatmap checkpoint was trained with old single-task "
                "conditioning. Retrain it with the current multitask trainer."
            )

        if "patch_backbone" not in payload:
            raise ValueError("RVT2Heatmap checkpoint is missing patch_backbone.")
        backbone_type = str(payload["patch_backbone"])
        dino_ckpt = (
            Path(self.config["dino_ckpt"]).expanduser()
            if backbone_type == "dino"
            else None
        )
        self.patch_backbone = PatchFeatureBackbone(
            backbone_type=backbone_type,
            dino_ckpt_path=dino_ckpt,
            image_size=self.vit_in,
            patch_size=self.patch_size,
            conv_dim=int(self.config["conv_patch_dim"]),
        )
        backbone_state_dict = payload.get("patch_backbone_state_dict")
        if backbone_state_dict is None:
            if backbone_type != "dino":
                raise ValueError(
                    "RVT2Heatmap checkpoint is missing patch_backbone_state_dict."
                )
            # PatchFeatureBackbone already loaded the frozen DINOv3 weights
            # from dino_ckpt_path above; a "dino" backbone never trains, so a
            # released checkpoint may omit this key entirely.
        else:
            self.patch_backbone.load_state_dict(backbone_state_dict, strict=True)
        head_config = dict(payload["head_config"])
        head_config["patch_dim"] = int(self.patch_backbone.output_dim)
        self.head = PatchActivationHead(**head_config)
        self.head.load_state_dict(payload["head_state_dict"], strict=True)
        self.head.eval()
        self.patch_backbone.eval()

        self.register_buffer(
            "gripper_mean",
            torch.as_tensor(np.asarray(payload["gripper_mean"], dtype=np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "gripper_std",
            torch.as_tensor(np.asarray(payload["gripper_std"], dtype=np.float32)),
            persistent=False,
        )

    @torch.no_grad()
    def _infer_heatmap_center(
        self,
        *,
        image_vit: torch.Tensor,
        composer_in: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "raw_gripper_opening" not in composer_in:
            raise KeyError(
                "composer_in must include raw_gripper_opening for RVT2Heatmap cropping"
            )
        if "robot_id" not in composer_in or "task_language_tokens" not in composer_in:
            raise KeyError(
                "composer_in must include robot_id and task_language_tokens for "
                "RVT2Heatmap cropping"
            )

        device = image_vit.device
        self.patch_backbone.eval()
        self.head.eval()

        gripper = composer_in["raw_gripper_opening"].to(
            device=device,
            dtype=image_vit.dtype,
        )
        mean = self.gripper_mean.to(device=device, dtype=image_vit.dtype)
        std = self.gripper_std.to(device=device, dtype=image_vit.dtype)
        gripper = (gripper - mean) / std.clamp_min(1e-6)

        task_embedding = composer_in.get("task_embedding")
        if task_embedding is None:
            task_embedding = torch.empty(
                image_vit.shape[0],
                0,
                device=device,
                dtype=image_vit.dtype,
            )
        logits = self.head(
            self.patch_backbone(image_vit),
            gripper,
            task_embedding.to(device=device, dtype=image_vit.dtype),
            composer_in["robot_id"].to(device=device, dtype=image_vit.dtype),
            eef_pos=(
                None
                if "eef_pos" not in composer_in
                else composer_in["eef_pos"].to(device=device, dtype=image_vit.dtype)
            ),
            task_language_tokens=(
                composer_in["task_language_tokens"].to(
                    device=device,
                    dtype=image_vit.dtype,
                )
            ),
        )
        probs = F.softmax(logits, dim=1)
        s = int(self.grid_res)
        mask_grid = probs.reshape(-1, 1, s, s)

        pred = probs.argmax(dim=1)
        row = torch.div(pred, s, rounding_mode="floor").to(image_vit.dtype)
        col = (pred % s).to(image_vit.dtype)
        cell = float(self.vit_in) / float(s)
        cx = (col + 0.5) * cell
        cy = (row + 0.5) * cell
        return mask_grid, cx, cy

    def _visual_focus_from_center(
        self,
        *,
        mask_grid: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        zoom: float,
    ) -> VisualFocusPrediction:
        if float(zoom) <= 0.0:
            raise ValueError(f"RVT2Heatmap zoom must be > 0, got {zoom}")
        side = float(self.vit_in) / max(float(zoom), 1e-6)
        box_px = torch.stack(
            [
                cx - side * 0.5,
                cy - side * 0.5,
                cx + side * 0.5,
                cy + side * 0.5,
            ],
            dim=-1,
        )
        return VisualFocusPrediction(
            box_px=box_px.clamp(0.0, float(self.vit_in - 1)),
            mask_grid=mask_grid,
            heatmap=mask_grid,
            source="rvt2_heatmap",
            metadata={"zoom": float(zoom)},
        )

    @torch.no_grad()
    def infer_visual_focus(
        self,
        *,
        image_vit: torch.Tensor,
        composer_in: dict,
    ) -> VisualFocusPrediction:
        """Infer a fixed-zoom crop box from RVT2Heatmap patch activations."""
        mask_grid, cx, cy = self._infer_heatmap_center(
            image_vit=image_vit,
            composer_in=composer_in,
        )
        return self._visual_focus_from_center(
            mask_grid=mask_grid,
            cx=cx,
            cy=cy,
            zoom=self.zoom,
        )

    @torch.no_grad()
    def infer_visual_focuses(
        self,
        *,
        image_vit: torch.Tensor,
        composer_in: dict,
        zoom_by_source: dict[str, float],
    ) -> dict[str, VisualFocusPrediction]:
        """Infer multiple fixed-zoom boxes while reusing one heatmap forward pass."""
        mask_grid, cx, cy = self._infer_heatmap_center(
            image_vit=image_vit,
            composer_in=composer_in,
        )
        return {
            str(source): self._visual_focus_from_center(
                mask_grid=mask_grid,
                cx=cx,
                cy=cy,
                zoom=float(zoom),
            )
            for source, zoom in zoom_by_source.items()
        }

    @torch.no_grad()
    def predict_visual_focus(
        self,
        *,
        image: torch.Tensor,
        composer_in: Mapping[str, torch.Tensor],
        view_name: str | None = None,
    ) -> VisualFocusPrediction:
        """Predict RVT2Heatmap focus for one image/view.

        ``view_name`` is accepted for interface symmetry; the current checkpoint is
        trained on agentview.
        """
        _ = view_name
        return self.infer_visual_focus(
            image_vit=image,
            composer_in=dict(composer_in),
        )
