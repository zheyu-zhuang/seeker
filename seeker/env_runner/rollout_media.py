"""Media helpers for rollout evaluation."""

from __future__ import annotations

import math
import os
from typing import Optional

import cv2
import numpy as np
import torch

from seeker.policy.base_image_policy import BaseImagePolicy


def extract_agentview_start_frames(obs: dict) -> np.ndarray:
    """Return reset-time agentview frames as HWC uint8 images."""
    agentview = obs["agentview_image"]
    if agentview.ndim != 5:
        raise ValueError(
            "Expected agentview_image to have shape [B, T, C, H, W], "
            f"got {agentview.shape}"
        )

    # MultiStepWrapper repeats the reset observation across the time axis when needed.
    frames = agentview[:, 0]
    frames = np.moveaxis(frames, 1, -1)
    frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    return frames


def _annotate_rollout_tile(
    image: np.ndarray,
    *,
    success: bool,
    label: str,
    fail_fade: float = 0.35,
) -> np.ndarray:
    out = image.copy()
    if not success:
        white = np.full_like(out, 255)
        out = cv2.addWeighted(out, fail_fade, white, 1.0 - fail_fade, 0.0)

    color = (60, 180, 75) if success else (220, 80, 80)
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness=4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    max_text_w = max(w - 16, 1)
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    if tw > max_text_w:
        scale = max(0.3, scale * max_text_w / float(tw))
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    x0, y0 = 8, 8
    cv2.rectangle(
        out,
        (x0 - 4, y0 - 4),
        (x0 + tw + 4, y0 + th + baseline + 4),
        color,
        thickness=-1,
    )
    cv2.putText(
        out,
        label,
        (x0, y0 + th),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return out


def save_image_grid(
    images: list[np.ndarray],
    output_path: str,
    pad: int = 2,
    *,
    success_flags: Optional[list[bool]] = None,
    labels: Optional[list[str]] = None,
) -> None:
    """Save a near-square image grid from HWC uint8 RGB frames."""
    if len(images) == 0:
        return

    first = images[0]
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {first.shape}")
    if (success_flags is None) != (labels is None):
        raise ValueError("success_flags and labels must be provided together")
    if success_flags is not None and (
        len(success_flags) != len(images) or len(labels) != len(images)
    ):
        raise ValueError("success_flags and labels must match images length")

    h, w, c = first.shape
    if len(images) == 50:
        rows, cols = 5, 10
    else:
        cols = math.ceil(math.sqrt(len(images)))
        rows = math.ceil(len(images) / cols)

    grid_h = rows * h + pad * max(rows - 1, 0)
    grid_w = cols * w + pad * max(cols - 1, 0)
    grid = np.zeros((grid_h, grid_w, c), dtype=np.uint8)

    for idx, image in enumerate(images):
        if image.shape != first.shape:
            raise ValueError(
                f"All grid images must have the same shape. "
                f"Expected {first.shape}, got {image.shape} at index {idx}."
            )
        row = idx // cols
        col = idx % cols
        y0 = row * (h + pad)
        x0 = col * (w + pad)
        if success_flags is not None and labels is not None:
            image = _annotate_rollout_tile(
                image,
                success=bool(success_flags[idx]),
                label=str(labels[idx]),
            )
        grid[y0 : y0 + h, x0 : x0 + w] = image

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    ok = cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to save image grid to {output_path}")


def rollout_env_grid_filename(env_seeds: list[int]) -> str:
    """Return a stable filename keyed by the ordered rollout env seed range."""
    if len(env_seeds) == 0:
        raise ValueError("env_seeds must be non-empty")
    return f"rollout_env_seed_{env_seeds[0]}-{env_seeds[-1]}.png"


def rollout_snapshot_label(epoch: Optional[int]) -> str:
    """Return the label used for an epoch-associated rollout snapshot."""
    if epoch is None:
        return "epoch_unknown_rollout"
    return f"epoch_{int(epoch):04d}_rollout"


def rollout_snapshot_filename(*, epoch: Optional[int], env_seeds: list[int]) -> str:
    """Return the rollout snapshot grid filename for an epoch and seed range."""
    return f"{rollout_snapshot_label(epoch)}_{rollout_env_grid_filename(env_seeds)}"


def rollout_video_filename(*, epoch: Optional[int], prefix: str, seed: int) -> str:
    """Return a stable rollout video filename keyed by split, seed, and epoch."""
    split = str(prefix).strip("/") or "rollout"
    epoch_name = "unknown" if epoch is None else f"{int(epoch):04d}"
    return f"epoch_{epoch_name}_{split}_seed_{int(seed)}.mp4"


def extract_policy_prediction_video_boxes(
    policy: BaseImagePolicy,
    *,
    n_envs: int,
) -> list[list[dict]]:
    """Return per-env predicted focus overlays recorded by the obs encoder."""
    obs_encoder = getattr(policy, "obs_encoder", None)
    last_video_items = []
    encoder_items = getattr(obs_encoder, "last_video_boxes", None)
    if encoder_items:
        for item in encoder_items:
            last_video_items.append(item)

    per_env = [[] for _ in range(n_envs)]
    if not last_video_items:
        return per_env

    for item in last_video_items:
        if not isinstance(item, dict):
            continue
        box_px = item.get("box_px")
        points_px = item.get("points_px")
        if box_px is None and points_px is None:
            continue

        box_arr = _to_numpy_or_none(box_px)
        points_arr = _to_numpy_or_none(points_px)
        batch_count = [
            int(arr.shape[0])
            for arr in (box_arr, points_arr)
            if arr is not None and arr.ndim > 0
        ]
        if not batch_count:
            continue

        count = min(n_envs, *batch_count)
        for env_idx in range(count):
            overlay = {
                "source": str(item.get("source", "pred")),
                "view": str(item.get("view", "agentview")),
                "source_size": int(item.get("source_size", 224)),
            }
            if box_arr is not None:
                overlay["box_px"] = (
                    box_arr[env_idx].astype(np.float32, copy=False).tolist()
                )
            if points_arr is not None:
                overlay["points_px"] = (
                    points_arr[env_idx].astype(np.float32, copy=False).tolist()
                )
                if "mean_point_index" in item:
                    overlay["mean_point_index"] = int(item["mean_point_index"])
            per_env[env_idx].append(overlay)
    return per_env


def _to_numpy_or_none(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)
