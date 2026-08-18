"""Visualization helpers for attention maps, trajectories, and diagnostics."""

import os
import textwrap
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision import utils as vutils
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

from seeker.util.image_ops import image_to_float01


def plot_switch_points(
    scores: Union[np.ndarray, torch.Tensor],
    switch_idx: Sequence[int],
    images: Optional[Union[np.ndarray, torch.Tensor]] = None,
    *,
    episode_index: Optional[int] = None,
    include_start: bool = True,
):
    """
    Plot temporal change-point scores and optionally show selected key frames.

    Args:
        scores: Score vector with shape [T].
        switch_idx: Candidate change-point indices.
        images: Optional frame tensor/array with shape [T,3,H,W] or [T,H,W,3].
        episode_index: Optional episode id for title.
        include_start: If True, force index 0 to be included.

    Returns:
        (fig, switch_idx_np)
          fig: matplotlib figure.
          switch_idx_np: validated/sorted switch indices used in plot.
    """

    scores_np = np.asarray(scores)
    if scores_np.ndim != 1:
        raise ValueError(f"scores must be 1D [T], got shape {scores_np.shape}")

    T = scores_np.shape[0]
    idx = np.array(sorted(set(int(i) for i in switch_idx)), dtype=int)
    idx = idx[(idx >= 0) & (idx < T)]
    if include_start and (idx.size == 0 or idx[0] != 0):
        idx = np.concatenate([np.array([0], dtype=int), idx])

    title = "Context-change score"
    if episode_index is not None:
        title = f"{title} (episode {episode_index})"

    if images is None:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(scores_np, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Score")
        if idx.size:
            ax.scatter(idx, scores_np[idx], zorder=3)
            for i in idx:
                ax.axvline(i, linestyle="--", linewidth=1)
        return fig, idx

    if isinstance(images, torch.Tensor):
        x = images.detach().cpu()
        if x.ndim != 4:
            raise ValueError(f"images tensor must be [T,3,H,W], got {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"images channel dim must be 3, got {x.shape[1]}")
        x = image_to_float01(x, source="auto")
        images_np = x.permute(0, 2, 3, 1).numpy()
    else:
        images_np = np.asarray(images)
        if images_np.ndim != 4:
            raise ValueError(
                f"images array must be [T,H,W,3] or [T,3,H,W], got {images_np.shape}"
            )
        if images_np.shape[-1] == 3:
            pass
        elif images_np.shape[1] == 3:
            images_np = np.transpose(images_np, (0, 2, 3, 1))
        else:
            raise ValueError(
                f"images array must have 3 channels in dim 1 or -1, got {images_np.shape}"
            )

    ncols = max(len(idx), 1)
    fig = plt.figure(figsize=(4 * max(ncols, 3), 6))
    gs = fig.add_gridspec(
        2, ncols, height_ratios=[2.0, 3.0], hspace=0.45, wspace=0.05
    )

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(scores_np, linewidth=1.5)
    ax0.set_title(title)
    ax0.set_xlabel("Frame")
    ax0.set_ylabel("Score")
    if idx.size:
        ax0.scatter(idx, scores_np[idx], zorder=3)
        for i in idx:
            ax0.axvline(i, linestyle="--", linewidth=1)

    for j, frame_idx in enumerate(idx):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(images_np[frame_idx])
        ax.set_title(f"t={frame_idx}\n{scores_np[frame_idx]:.3f}", fontsize=10)
        ax.axis("off")

    return fig, idx


def overlay_boxes_on_images(
    images: torch.Tensor,
    boxes: torch.Tensor,
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 2,
) -> torch.Tensor:
    """
    Draw boxes on images.

    images: [N, 3, H, W] in [0,1]
    boxes:  [N, 4] (x1, y1, x2, y2) in pixel coords
    """
    if images is None or boxes is None:
        return images

    N, _, H, W = images.shape
    if boxes.shape[0] != N:
        raise ValueError(f"boxes batch mismatch: images N={N}, boxes N={boxes.shape[0]}")

    out = images.clone()
    line_color = out.new_tensor(color).view(3, 1) / 255.0

    for i in range(N):
        x1, y1, x2, y2 = boxes[i].round().to(torch.int64).tolist()
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))
        if x2 < x1 or y2 < y1:
            continue

        for k in range(max(1, int(width))):
            xl = max(0, x1 - k)
            xr = min(W - 1, x2 + k)
            yt = max(0, y1 - k)
            yb = min(H - 1, y2 + k)

            out[i, :, yt, xl : xr + 1] = line_color
            out[i, :, yb, xl : xr + 1] = line_color
            out[i, :, yt : yb + 1, xl] = line_color
            out[i, :, yt : yb + 1, xr] = line_color

    return out


def overlay_masks_on_images(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    blackout: bool = False,
    mask_interp: str = "nearest",
) -> torch.Tensor:
    """
    Blend mask onto image as red heatmaps.

    image: [N, 3, H, W] in [0,1]
    mask:  [N, 1, h, w] (will be resized & normalized)
    """
    if image is None or mask is None:
        return image

    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"image must be [N,3,H,W] or [3,H,W], got shape {tuple(image.shape)}")

    N, C, H, W = image.shape
    if C != 3:
        raise ValueError(f"image channel count must be 3, got {C}")

    # Accept [N,1,h,w], [N,h,w], [1,h,w], or [h,w].
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    elif mask.dim() == 3:
        if mask.shape[0] == N:
            mask = mask.unsqueeze(1)  # [N,1,h,w]
        elif N == 1:
            mask = mask.unsqueeze(0)  # [1,1,h,w] for [1,h,w]
        else:
            raise ValueError(
                f"mask batch mismatch: image N={N}, mask shape={tuple(mask.shape)}"
            )
    elif mask.dim() != 4:
        raise ValueError(
            f"mask must be [N,1,h,w], [N,h,w], [1,h,w], or [h,w]; got {tuple(mask.shape)}"
        )

    if mask.shape[0] == 1 and N > 1:
        mask = mask.expand(N, -1, -1, -1)
    if mask.shape[0] != N:
        raise ValueError(f"mask batch mismatch: image N={N}, mask N={mask.shape[0]}")
    if mask.shape[1] != 1:
        # Collapse multi-channel masks to one map.
        mask = mask.mean(dim=1, keepdim=True)

    _, _, h, w = mask.shape

    if (h, w) != (H, W):
        mask = F.interpolate(mask.float(), size=(H, W), mode=mask_interp)
    else:
        mask = mask.float()
    mask = mask.clamp(0, 1)
    # duplicate to 3 channels
    m_rgb = mask.repeat(1, 3, 1, 1)

    if not blackout:
        return (1 - alpha) * image + alpha * m_rgb

    gray = 0.3 * torch.ones_like(image)

    return m_rgb * image + (1 - m_rgb) * gray


def draw_scores_on_grid(
    grid: torch.Tensor,
    scores: torch.Tensor,
    tile_h: int,
    tile_w: int,
    nrow: int,
    padding: int = 2,
) -> torch.Tensor:
    """
    Draw scalar scores onto a grid image.

    grid:   [3, H_tot, W_tot]
    scores: [rows, cols]  (rows = num_tiles / nrow, cols = nrow)
    """
    grid_cpu = grid.detach().cpu()
    pil_img = TF.to_pil_image(grid_cpu)
    draw = ImageDraw.Draw(pil_img)

    rows, cols = scores.shape
    for r in range(rows):
        for c in range(cols):
            val = scores[r, c].item()
            text = f"{val:.3f}"
            x = padding + c * (tile_w + padding) + 2
            y = padding + r * (tile_h + padding) + 2
            draw.text((x, y), text, fill=(255, 255, 255))

    return TF.to_tensor(pil_img).to(grid.device)


# ---------------------- top-level visualize (handles temporal) ---------------------- #


def _recover_bt(t: torch.Tensor, temporal_dim: int) -> Tuple[torch.Tensor, int, int]:
    """
    Recover [B, T, ...] from either [B, T, ...] or [B*T, ...].
    """
    if t is None:
        return None, 0, 0

    if t.dim() in (2, 3) and temporal_dim is not None and t.shape[1] == temporal_dim:
        # Already [B, T, ...] for low-dim tensors such as boxes/scores.
        B, T = t.shape[0], t.shape[1]
        return t, B, T

    if t.dim() >= 5:  # e.g. [B, T, C, H, W] or [B, T, 1, H, W]
        B, T = t.shape[0], t.shape[1]
        if temporal_dim is not None:
            assert T == temporal_dim, f"T={T}, expected {temporal_dim}"
        return t, B, T

    # flattened: [B*T, ...]
    BT = t.shape[0]
    assert temporal_dim is not None, "temporal_dim required for flattened inputs."
    assert BT % temporal_dim == 0, f"BT={BT} not divisible by T={temporal_dim}"
    B = BT // temporal_dim
    new_shape = (B, temporal_dim) + t.shape[1:]
    return t.view(*new_shape), B, temporal_dim


def visualize(
    views: Sequence[
        Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ]
    ],
    temporal_dim: int = 1,
    save_dir: str = None,
    step: Optional[int] = None,
    num_viz: int = 16,
    padding: int = 2,
    text: Optional[str] = None,
    step_chunk_size: int = 10_000,
) -> torch.Tensor:
    """
    views: list of (img, mask, box, score) tuples

      img:   [B, T, 3, H, W] or [B*T, 3, H, W]
      mask:  [B, T, 1, H, W] or [B*T, 1, H, W] or None
      box:   [B, T, 4] or [B*T, 4] or None
      score: [B, T, 1] or [B*T, 1] or None

    Temporal dimension is only handled here; low-level helpers see [N, ...].
    """
    if save_dir is not None:
        if step is not None and step_chunk_size > 0:
            chunk_start = (int(step) // step_chunk_size) * step_chunk_size
            chunk_end = chunk_start + step_chunk_size - 1
            save_dir = os.path.join(save_dir, f"steps_{chunk_start:06d}_{chunk_end:06d}")
        os.makedirs(save_dir, exist_ok=True)
        fname = "viz.png" if step is None else f"train_step_{step}_viz.png"
        save_path = os.path.join(save_dir, fname)

    assert len(views) > 0

    overlay_rows = []  # list of [N, 3, H, W]
    score_rows = []  # list of [N]
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    # We assume all views share the same B, T, H, W
    for img, mask, box, score in views:
        # recover [B, T, ...]
        img, B, T = _recover_bt(img, temporal_dim)
        mask, _, _ = (
            _recover_bt(mask, temporal_dim) if mask is not None else (None, 0, 0)
        )

        if mask is not None:
            m_max = mask.amax(dim=(-4, -3, -2, -1), keepdim=True) + 1e-8
            mask = (mask / m_max).clamp(0, 1)

        box_list = None
        if box is not None:
            box_list = box if isinstance(box, list) else [box]
            box_list = [(_recover_bt(b, temporal_dim)[0]) for b in box_list]
        score, _, _ = (
            _recover_bt(score, temporal_dim) if score is not None else (None, 0, 0)
        )

        B = min(B, num_viz)  # limit columns
        img = img[:B]
        text_items = None
        if text is not None:
            if isinstance(text, str):
                text_items = [text] * B
            else:
                text_items = list(text)[:B]
        if mask is not None:
            mask = mask[:B]
        if box_list is not None:
            box_list = [b[:B] for b in box_list]
        if score is not None:
            score = score[:B]

        _, _, _, H, W = img.shape

        # iterate over time, build rows
        for t in range(T):
            img_t = image_to_float01(img[:, t], source="imagenet")  # [B, 3, H, W]

            if box_list is not None:
                for i, box_ in enumerate(box_list):
                    box_t = box_[:, t]  # [B, 4]
                    img_t = overlay_boxes_on_images(
                        img_t, box_t, color=colours[i % 4], width=2
                    )

            if mask is not None:
                mask_t = mask[:, t]  # [B, 1, h, w]
                img_t = overlay_masks_on_images(img_t, mask_t)

            if text_items is not None:
                # draw text on each image
                img_t_cpu = img_t.detach().cpu()
                pil_imgs = [
                    TF.to_pil_image(img_t_cpu[i]) for i in range(img_t_cpu.size(0))
                ]
                draw_imgs = []
                max_chars = 50  # wrap width, adjust up/down as needed
                for i, pil_img in enumerate(pil_imgs):
                    draw = ImageDraw.Draw(pil_img)
                    wrapped = textwrap.fill(str(text_items[i]), width=max_chars)
                    draw.multiline_text(
                        (5, 5),
                        wrapped,
                        fill=(255, 255, 255),
                        spacing=2,
                    )
                    draw_imgs.append(TF.to_tensor(pil_img))
                img_t = torch.stack(draw_imgs, dim=0).to(img_t.device)

            overlay_rows.append(img_t)

            if score is not None:
                score_t = score[:, t, 0]  # [B]
                score_rows.append(score_t)
            else:
                score_ = torch.ones(img_t.size(0), device=img_t.device)
                score_rows.append(score_)

    # Stack rows and make grid
    rows = len(overlay_rows)
    N = overlay_rows[0].size(0)
    H, W = overlay_rows[0].shape[-2:]

    overlay = torch.stack(overlay_rows, dim=0)  # [rows, N, 3, H, W]
    overlay_flat = overlay.view(rows * N, 3, H, W)  # [rows*N, 3, H, W]
    scores_mat = torch.stack(score_rows, dim=0)  # [rows, N]

    grid = vutils.make_grid(
        overlay_flat,
        nrow=N,
        normalize=False,
        scale_each=False,
        padding=padding,
    )

    if save_dir is not None:
        vutils.save_image(grid, save_path)
    return grid


def denorm_image(x: torch.Tensor) -> torch.Tensor:
    if x.min().item() >= 0:
        return x
    mean = x.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (x * std + mean).clamp(0, 1)


def _magma_lut(device):
    # lightweight perceptual-ish LUT (no matplotlib)
    x = torch.linspace(0, 1, 256, device=device)
    r = torch.clamp(1.5 * x, 0, 1)
    g = torch.clamp(1.5 * (x - 0.2), 0, 1)
    b = torch.clamp(1.5 * (x - 0.5), 0, 1)
    return torch.stack([r, g, b], dim=-1)  # [256,3]


def _pad_to_even_hw_u8(frame_hwc_u8: torch.Tensor) -> torch.Tensor:
    """frame_hwc_u8: [H,W,3] uint8"""
    assert (
        frame_hwc_u8.dtype == torch.uint8
        and frame_hwc_u8.ndim == 3
        and frame_hwc_u8.shape[-1] == 3
    )
    H, W, _ = frame_hwc_u8.shape
    pad_h = (2 - (H % 2)) % 2
    pad_w = (2 - (W % 2)) % 2
    if pad_h == 0 and pad_w == 0:
        return frame_hwc_u8
    return F.pad(frame_hwc_u8, (0, 0, 0, pad_w, 0, pad_h), value=0)


@torch.no_grad()
def save_attention_heads_video(
    images: torch.Tensor,  # [T,3,H,W]
    attn: torch.Tensor,  # [T,H,1,196] (or [T,H,1,197] w/ CLS)
    head_score: torch.Tensor,  # [T,H,1,1] or [T,H]
    save_path: str,
    *,
    grid_hw: int = 14,
    fps: int = 30,
    alpha: float = 0.55,
    tile_scale: float = 0.45,
    bar_h: int = 28,
    bar_margin: int = 6,
    gap: int = 10,
    top_text_h: int = 0,
    font_size: int = 14,
    normalize_sum1: bool = True,
    minmax_per_head: bool = True,
    score_softmax_over_heads: bool = False,
    save_frames_every=10,
):
    HEAD_COLORS = [
        (0, 114, 178),  # blue
        (230, 159, 0),  # orange
        (0, 158, 115),  # green
        (204, 121, 167),  # purple
        (213, 94, 0),  # vermillion
        (86, 180, 233),  # sky blue
    ]

    assert (
        images.ndim == 4 and images.shape[1] == 3
    ), f"images must be [T,3,H,W], got {tuple(images.shape)}"
    assert (
        attn.ndim == 4 and attn.shape[2] == 1
    ), f"attn must be [T,H,1,N], got {tuple(attn.shape)}"
    T, _, H, W = images.shape
    assert attn.shape[0] == T, "attn T mismatch"

    Hh = attn.shape[1]
    N = attn.shape[-1]
    assert Hh == 6, f"This layout expects 6 heads; got H={Hh}"

    # head_score -> [T,H]
    if head_score.ndim == 4:
        assert head_score.shape[:2] == (T, Hh) and head_score.shape[2:] == (
            1,
            1,
        ), f"head_score expected [T,H,1,1], got {tuple(head_score.shape)}"
        s = head_score.view(T, Hh).detach().float()
    elif head_score.ndim == 2:
        assert head_score.shape == (
            T,
            Hh,
        ), f"head_score expected [T,H], got {tuple(head_score.shape)}"
        s = head_score.detach().float()
    else:
        raise ValueError(f"Unexpected head_score shape {tuple(head_score.shape)}")

    if score_softmax_over_heads:
        w_bar = torch.softmax(s, dim=1)
    else:
        w_bar = s.clamp(min=0)
        w_bar = w_bar / (w_bar.sum(dim=1, keepdim=True) + 1e-8)  # sums to 1

    # attn -> [T,H,N] (drop CLS if needed)
    a = attn[:, :, 0, :].detach().float()  # [T,H,N]
    if N == grid_hw * grid_hw + 1:
        a = a[:, :, 1:]
        N = a.shape[-1]
    assert N == grid_hw * grid_hw, f"N={N} not compatible with grid_hw={grid_hw}"

    if normalize_sum1:
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)

    # attention grid -> upsample to image res using NEAREST
    a_grid = a.view(T, Hh, grid_hw, grid_hw)  # [T,H,gh,gw]
    a_up = F.interpolate(
        a_grid.reshape(T * Hh, 1, grid_hw, grid_hw),
        size=(H, W),
        mode="nearest",
    ).reshape(T, Hh, 1, H, W)

    if minmax_per_head:
        vmin = a_up.amin(dim=(-2, -1), keepdim=True)
        vmax = a_up.amax(dim=(-2, -1), keepdim=True)
        a_vis = (a_up - vmin) / (vmax - vmin + 1e-8)
    else:
        a_vis = a_up.clamp(0, 1)

    base = image_to_float01(images, source="imagenet")  # [T,3,H,W]
    lut = _magma_lut(images.device)  # [256,3]

    tile_h = max(8, int(round(H * tile_scale)))
    tile_w = max(8, int(round(W * tile_scale)))

    frame_w = Hh * tile_w + (Hh - 1) * gap
    frame_h = top_text_h + bar_h + tile_h

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    frame_dir = os.path.splitext(save_path)[0]
    should_dump_frames = save_frames_every is not None and int(save_frames_every) > 0
    if should_dump_frames:
        os.makedirs(frame_dir, exist_ok=True)

    frames_u8 = []
    for t in range(T):
        canvas = Image.new("RGB", (frame_w, frame_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        if top_text_h > 0:
            draw.text((6, 2), f"t={t}", fill=(0, 0, 0), font=font)

        y_bar0 = top_text_h
        y_img0 = top_text_h + bar_h

        for h in range(Hh):
            img = base[t]  # [3,H,W]
            m = a_vis[t, h, 0]  # [H,W] in [0,1]

            idx = (m * 255.0).round().to(torch.long).clamp(0, 255)
            heat = lut[idx].permute(2, 0, 1).contiguous()  # [3,H,W]
            overlay = ((1 - alpha) * img + alpha * heat).clamp(0, 1)

            # to PIL, then resize tile with NEAREST (keeps blocky attention look)
            u8 = (
                (overlay * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            )
            pil_tile = Image.fromarray(u8).resize(
                (tile_w, tile_h), resample=Image.NEAREST
            )

            x0 = h * (tile_w + gap)

            draw.rectangle(
                [x0, y_bar0, x0 + tile_w, y_bar0 + bar_h], fill=(255, 255, 255)
            )
            draw.rectangle(
                [x0, y_bar0, x0 + tile_w, y_bar0 + bar_h],
                outline=(230, 230, 230),
                width=1,
            )

            w = float(w_bar[t, h].item())  # in [0,1]
            inner_w = max(1, tile_w - 2 * bar_margin)
            inner_h = max(1, bar_h - 2 * bar_margin)
            fill_w = int(round(inner_w * w))

            bx0 = x0 + bar_margin
            by0 = y_bar0 + bar_margin
            bx1 = bx0 + inner_w
            by1 = by0 + inner_h

            draw.rectangle(
                [bx0, by0, bx1, by1],
                fill=(245, 245, 245),
                outline=(210, 210, 210),
                width=1,
            )
            if fill_w > 0:
                draw.rectangle(
                    [bx0, by0, bx0 + fill_w, by1], fill=HEAD_COLORS[h], outline=None
                )

            pct = int(round(100 * w))
            label = f"{pct:02d}%"
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]

            pad = 2
            tx = bx1 - tw - pad

            raise_px = 2
            ty = by0 + (inner_h - th) // 2 - raise_px
            ty = max(by0, min(ty, by1 - th))  # clamp

            draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

            canvas.paste(pil_tile, (x0, y_img0))
            draw.rectangle(
                [x0, y_img0, x0 + tile_w - 1, y_img0 + tile_h - 1],
                outline=(230, 230, 230),
                width=1,
            )

        frame = torch.from_numpy(np.array(canvas)).to(torch.uint8)  # [H,W,3]
        frame = _pad_to_even_hw_u8(frame)
        frames_u8.append(frame)

        if should_dump_frames and (t % int(save_frames_every) == 0):
            out_path = os.path.join(frame_dir, f"{t:06d}.png")
            torchvision.io.write_png(frames_u8[-1].permute(2, 0, 1).cpu(), out_path)

    vid = torch.stack(frames_u8, dim=0)  # [T,H,W,3] uint8
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torchvision.io.write_video(save_path, vid, fps=fps)


def visualize_trajectory(
    images: torch.Tensor,  # [T,3,H,W] (agentview)
    *,
    mask: Optional[torch.Tensor] = None,  # [T,1,H,W] (agentview)
    boxes: Optional[List[torch.Tensor]] = None,  # list of [T,4] (agentview)
    eih_images: Optional[torch.Tensor] = None,  # [T,3,H,W]
    eih_mask: Optional[torch.Tensor] = None,  # [T,1,H,W]
    eih_boxes: Optional[List[torch.Tensor]] = None,  # list of [T,4]
    gripper_opening: Optional[torch.Tensor] = None,  # [T] or [T,1], in [-1,1]
    save_dir: Optional[str] = None,
    step: Optional[int] = None,
    prefix: str = "frame",
    text: Optional[Union[str, Sequence[str]]] = None,
    draw_gripper_bar: bool = True,
    box_smoothing: Optional[float] = None,
    mask_smoothing: Optional[float] = None,
    box_padding: int = 8,
    mask_interp: str = "nearest",
    blackout: bool = False,
    save_video: bool = False,
):
    assert (
        images.dim() == 4 and images.shape[1] == 3
    ), f"images must be [T,3,H,W], got {tuple(images.shape)}"

    device = images.device
    T, _, H, W = images.shape

    if eih_images is not None:
        assert (
            eih_images.dim() == 4 and eih_images.shape[1] == 3
        ), f"eih_images must be [T,3,H,W], got {tuple(eih_images.shape)}"
        assert eih_images.shape[0] == T, "eih_images must have same T as images"

    images_vis = image_to_float01(images, source="imagenet")
    eih_images_vis = None
    if eih_images is not None:
        eih_images_vis = image_to_float01(eih_images, source="imagenet")

    def _norm_and_smooth_mask(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if m is None:
            return None
        assert (
            m.shape[0] == T and m.shape[1] == 1
        ), f"mask must be [T,1,H,W], got {tuple(m.shape)}"
        m_max = m.amax(dim=(1, 2, 3), keepdim=True) + 1e-8
        m = (m / m_max).clamp(0, 1)
        if mask_smoothing is not None:
            m = smooth_mask(m, momentum=mask_smoothing)
        return m

    def _validate_and_smooth_boxes(
        bs: Optional[List[torch.Tensor]],
    ) -> Optional[List[torch.Tensor]]:
        if bs is None:
            return None
        for b in bs:
            assert b.shape == (T, 4), f"each box must be [T,4], got {tuple(b.shape)}"
        if box_smoothing is not None:
            pad_box = torch.tensor(
                [-box_padding, -box_padding, box_padding, box_padding],
                device=device,
                dtype=bs[0].dtype,
            )
            bs = [b + pad_box for b in bs]
            bs = [smooth_box(b, box_smoothing) for b in bs]
        return bs

    def _make_vpad(w, dtype, h=12):
        pad = torch.empty((1, 3, h, w), device=device, dtype=dtype)
        pad[:, 0].fill_(pad_rgb[0])
        pad[:, 1].fill_(pad_rgb[1])
        pad[:, 2].fill_(pad_rgb[2])
        return pad

    def _make_row(left, right):
        # left/right: [1,3,H,W]
        return torch.cat(
            [left, _make_pad(H, left.dtype), right], dim=-1
        )  # width concat

    mask = _norm_and_smooth_mask(mask)
    eih_mask = _norm_and_smooth_mask(eih_mask)

    boxes = _validate_and_smooth_boxes(boxes)
    eih_boxes = _validate_and_smooth_boxes(eih_boxes)

    if gripper_opening is not None:
        if gripper_opening.dim() == 1:
            gripper_opening = gripper_opening[:, None]
        assert gripper_opening.shape == (T, 1), "invalid gripper_opening shape"
        gripper_opening = gripper_opening.clamp(-1.0, 1.0)

    # normalize text to per-frame list
    if isinstance(text, str):
        text_list = [text] * T
    elif text is None:
        text_list = None
    else:
        assert len(text) == T, f"text must be str or length T={T}, got {len(text)}"
        text_list = list(text)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    pad_px = 16
    pad_rgb = (0.0, 0.0, 0.0)

    def _make_pad(h, dtype):
        pad = torch.empty((1, 3, h, pad_px), device=device, dtype=dtype)
        pad[:, 0].fill_(pad_rgb[0])
        pad[:, 1].fill_(pad_rgb[1])
        pad[:, 2].fill_(pad_rgb[2])
        return pad

    video_frames = []
    for t in range(T):
        base = images_vis[t].unsqueeze(0)  # [1,3,H,W]

        left = base.clone()
        if boxes is not None:
            for i, b in enumerate(boxes):
                left = overlay_boxes_on_images(
                    left,
                    b[t].unsqueeze(0),
                    color=colours[i % len(colours)],
                    width=2,
                )

        right = base.clone()
        if mask is not None:
            right = overlay_masks_on_images(
                right,
                mask[t].unsqueeze(0),
                mask_interp=mask_interp,
                blackout=blackout,
            )

        need_left_pil = (text_list is not None) or (
            draw_gripper_bar and gripper_opening is not None
        )
        if need_left_pil:
            pil = TF.to_pil_image(left[0].detach().cpu())
            draw = ImageDraw.Draw(pil)

            if text_list is not None:
                wrapped = textwrap.fill(text_list[t], width=60)
                draw.multiline_text((5, 5), wrapped, fill=(255, 255, 255), spacing=2)

            if draw_gripper_bar and (gripper_opening is not None):
                g = float(gripper_opening[t, 0])
                g = max(-1.0, min(1.0, g))
                grip_pct = (g + 1.0) * 50.0
                g01 = (g + 1.0) * 0.5

                bar_w = int(0.30 * pil.size[0])
                bar_h = 10
                margin = 5
                x0 = margin
                y0 = pil.size[1] - (bar_h + margin + 12)
                y0 = max(margin, y0)

                neutral = (120, 120, 120)
                draw.text(
                    (x0, max(margin, y0 - 12)), f"Grip    {grip_pct:.0f}%", fill=neutral
                )
                draw.rectangle(
                    [x0, y0, x0 + bar_w, y0 + bar_h], outline=neutral, width=1
                )
                draw.rectangle(
                    [x0, y0, x0 + int(bar_w * g01), y0 + bar_h], fill=neutral
                )

            left = TF.to_tensor(pil).to(device).unsqueeze(0)

        agent_row = _make_row(left, right)  # [1,3,H,Wrow]

        frame = agent_row
        if eih_images_vis is not None:
            ebase = eih_images_vis[t].unsqueeze(0)  # [1,3,H,W]

            eleft = ebase.clone()
            if eih_boxes is not None:
                for i, b in enumerate(eih_boxes):
                    eleft = overlay_boxes_on_images(
                        eleft,
                        b[t].unsqueeze(0),
                        color=colours[i % len(colours)],
                        width=2,
                    )

            eright = ebase.clone()
            if eih_mask is not None:
                eright = overlay_masks_on_images(
                    eright,
                    eih_mask[t].unsqueeze(0),
                    mask_interp=mask_interp,
                    blackout=blackout,
                )

            eih_row = _make_row(eleft, eright)  # [1,3,H,Wrow]

            assert (
                agent_row.shape[-1] == eih_row.shape[-1]
            ), f"Row width mismatch: agent={agent_row.shape[-1]} vs eih={eih_row.shape[-1]}"

            vpad = _make_vpad(agent_row.shape[-1], agent_row.dtype, h=12)
            frame = torch.cat([agent_row, vpad, eih_row], dim=-2)  # height concat

        if save_dir is not None and not save_video:
            fname = (
                f"{prefix}_t{t:04d}.png"
                if step is None
                else f"{prefix}_step{step}_t{t:04d}.png"
            )
            vutils.save_image(frame, os.path.join(save_dir, fname))

        if save_video:
            f = frame[0].detach().clamp(0, 1)
            f_u8 = (
                (f * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().contiguous()
            )
            f_u8 = _pad_to_even_hw_u8(f_u8)
            video_frames.append(f_u8)

    if save_video:
        assert save_dir is not None, "Provide save_dir when save_video=True"
        name = prefix if step is None else f"{prefix}_step{step}"
        video_path = os.path.join(save_dir, f"{name}.mp4")
        vid = torch.stack(video_frames, dim=0)  # [T,H,W,3] uint8
        torchvision.io.write_video(video_path, vid, fps=30)


def smooth_box(box: torch.Tensor, s: float) -> torch.Tensor:
    """
    box: [T,4] float tensor
    s in [0,1). 0 = no smoothing, 1: makes all boxes the same as first box.
    0 <= s < 1: exponential moving average smoothing.
    """
    if s is None or s <= 0.0:
        return box

    s = float(s)
    assert 0.0 <= s < 1.0, f"box_smoothing must be in [0,1], got {s}"

    x = box.float()
    out = x.clone()
    prev = x[0]
    out[0] = prev
    for t in range(1, x.shape[0]):
        prev = s * prev + (1 - s) * x[t]
        out[t] = prev
    return out


def smooth_mask(mask, momentum=0.2):
    """
    mask: [T,1,H,W] in [0,1]
    momentum: smaller = smoother
    """
    if momentum <= 0:
        return mask

    out = mask.clone()
    prev = mask[0]
    out[0] = prev

    for t in range(1, mask.shape[0]):
        prev = momentum * prev + (1 - momentum) * mask[t]
        out[t] = prev

    return out.clamp(0, 1)
