"""Rendering helpers for visual-focus debugging."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from seeker.util.image_ops import image_to_float01, resize_image


def patch_box(
    patch_idx: int,
    *,
    grid_size: int,
    image_size: int,
) -> tuple[int, int, int, int]:
    row = int(patch_idx) // int(grid_size)
    col = int(patch_idx) % int(grid_size)
    cell = float(image_size) / float(grid_size)
    x1 = int(round(col * cell))
    y1 = int(round(row * cell))
    x2 = int(round((col + 1) * cell)) - 1
    y2 = int(round((row + 1) * cell)) - 1
    return (
        max(0, min(x1, image_size - 1)),
        max(0, min(y1, image_size - 1)),
        max(0, min(x2, image_size - 1)),
        max(0, min(y2, image_size - 1)),
    )


def image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = (
        image_to_float01(image.detach().cpu(), source="auto")
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="RGB")


def overlay_heatmap(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    alpha: float,
) -> Image.Image:
    base = image_to_float01(image.detach().cpu(), source="auto")
    h, w = int(base.shape[-2]), int(base.shape[-1])
    heat = heatmap.detach().float().cpu()[None, None]
    heat = F.interpolate(heat, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    heat = heat / heat.max().clamp_min(1e-12)
    red = torch.zeros_like(base)
    red[0] = 1.0
    overlay = base * (1.0 - float(alpha) * heat[None]) + red * (
        float(alpha) * heat[None]
    )
    return image_tensor_to_pil(overlay)


def draw_patch_debug(
    *,
    image: torch.Tensor,
    prob_grid: torch.Tensor,
    target_patch: int,
    pred_patch: int,
    source_frame: int,
    target_frame: int,
    alpha: float,
) -> Image.Image:
    h, w = int(image.shape[-2]), int(image.shape[-1])
    if h != w:
        image = resize_image(
            image_to_float01(image, source="auto")[None],
            max(h, w),
        )[0]
        h = w = max(h, w)
    grid_size = int(prob_grid.shape[0])
    pil = overlay_heatmap(image, prob_grid, alpha=alpha)
    draw = ImageDraw.Draw(pil)

    gt_box = patch_box(target_patch, grid_size=grid_size, image_size=h)
    pred_box = patch_box(pred_patch, grid_size=grid_size, image_size=h)
    for k in range(3):
        gt = (gt_box[0] - k, gt_box[1] - k, gt_box[2] + k, gt_box[3] + k)
        pred = (
            pred_box[0] - k,
            pred_box[1] - k,
            pred_box[2] + k,
            pred_box[3] + k,
        )
        draw.rectangle(gt, outline=(0, 255, 0))
        draw.rectangle(pred, outline=(255, 220, 0))

    label = (
        f"src {source_frame} -> key {target_frame} | "
        f"pred {pred_patch} gt {target_patch}"
    )
    font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(label, font=font)
    draw.rectangle((0, 0, min(w - 1, text_w + 8), text_h + 6), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255), font=font)
    return pil
