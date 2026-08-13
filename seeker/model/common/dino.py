"""DINOv3 backbone loading helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from seeker.model.dinov3_core.make_dinov3_vits import dinov3_vits16plus


def load_frozen_dinov3_vits16plus(
    ckpt_path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> nn.Module:
    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            "DINOv3 checkpoint not found: "
            f"{ckpt_path}. "
            "Please follow 'https://github.com/facebookresearch/dinov3' "
            "to download the pretrained weights and provide the correct path "
            "in the config."
        )

    logging.getLogger("dinov3").setLevel(logging.WARNING)
    vit = dinov3_vits16plus(pretrained=False)
    vit.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    if device is not None:
        vit.to(device)
    vit.eval()
    for param in vit.parameters():
        param.requires_grad = False
    return vit
