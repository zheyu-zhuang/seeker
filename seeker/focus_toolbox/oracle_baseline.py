"""Oracle visual-focus box helpers shared by training, rollout, and eval."""

from __future__ import annotations

import torch


def target_patch_tight_boxes(
    target_patch_mask: torch.Tensor,
    *,
    image_size: int,
) -> torch.Tensor:
    """Return square tight boxes for each nonempty target mask frame."""
    if target_patch_mask.ndim != 3:
        raise ValueError(
            f"Expected target patch mask [T,S,S], got {target_patch_mask.shape}"
        )
    mask = target_patch_mask.bool()
    t, grid_h, grid_w = mask.shape
    boxes = torch.full((t, 4), float("nan"), dtype=torch.float32, device=mask.device)
    if t == 0:
        return boxes

    scale_x = float(image_size) / max(float(grid_w), 1.0)
    scale_y = float(image_size) / max(float(grid_h), 1.0)
    for frame in range(t):
        ys, xs = torch.nonzero(mask[frame], as_tuple=True)
        if ys.numel() == 0:
            continue

        x1_cell = int(xs.min().item())
        x2_cell = int(xs.max().item()) + 1
        y1_cell = int(ys.min().item())
        y2_cell = int(ys.max().item()) + 1

        width = x2_cell - x1_cell
        height = y2_cell - y1_cell
        side = max(width, height)
        side = min(side, grid_h, grid_w)

        x_pad = side - width
        y_pad = side - height
        x1_cell = x1_cell - x_pad // 2
        y1_cell = y1_cell - y_pad // 2
        x2_cell = x1_cell + side
        y2_cell = y1_cell + side

        if x1_cell < 0:
            x2_cell -= x1_cell
            x1_cell = 0
        if y1_cell < 0:
            y2_cell -= y1_cell
            y1_cell = 0
        if x2_cell > grid_w:
            x1_cell -= x2_cell - grid_w
            x2_cell = grid_w
        if y2_cell > grid_h:
            y1_cell -= y2_cell - grid_h
            y2_cell = grid_h

        x1 = torch.tensor(float(x1_cell) * scale_x, device=mask.device)
        y1 = torch.tensor(float(y1_cell) * scale_y, device=mask.device)
        x2 = torch.tensor(float(x2_cell) * scale_x, device=mask.device)
        y2 = torch.tensor(float(y2_cell) * scale_y, device=mask.device)
        boxes[frame] = torch.stack([x1, y1, x2, y2])
    return boxes


def replace_invalid_boxes(
    boxes: torch.Tensor,
    *,
    image_size: int,
) -> torch.Tensor:
    """Replace non-finite boxes with a crop-safe fallback for policy execution."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes [T,4], got {boxes.shape}")

    out = boxes.float().clone()
    valid = torch.isfinite(out).all(dim=1)
    if torch.all(valid):
        return out

    fallback_box = out.new_tensor([0.0, 0.0, float(image_size), float(image_size)])
    out[~valid] = fallback_box
    return out
