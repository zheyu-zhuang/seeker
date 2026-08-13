"""Shared image augmentation utilities."""

import os
from typing import Optional

import kornia as K
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from PIL import Image

from seeker.util.image_ops import normalize_imagenet


class CropRandomizer(nn.Module):
    """Random crop during training, center crop during eval."""

    def __init__(self, input_shape, crop_size):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.crop_size = int(crop_size)
        assert len(self.input_shape) == 3
        assert self.crop_size <= self.input_shape[1]
        assert self.crop_size <= self.input_shape[2]

    def forward(self, image: torch.Tensor, center_crop: bool = False) -> torch.Tensor:
        if not self.training or center_crop:
            return ttf.center_crop(image, [self.crop_size, self.crop_size])

        leading = image.shape[:-3]
        flat = image.reshape(-1, *image.shape[-3:])
        _, _, H, W = flat.shape
        max_top = H - self.crop_size
        max_left = W - self.crop_size
        tops = torch.randint(0, max_top + 1, (flat.shape[0],)).tolist()
        lefts = torch.randint(0, max_left + 1, (flat.shape[0],)).tolist()
        crops = [
            ttf.crop(x, int(top), int(left), self.crop_size, self.crop_size)
            for x, top, left in zip(flat, tops, lefts)
        ]
        return torch.stack(crops, dim=0).reshape(
            *leading,
            image.shape[-3],
            self.crop_size,
            self.crop_size,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(input_shape={self.input_shape}, crop_size={self.crop_size})"
        )


class BackgroundOverlay(nn.Module):
    """Blend images with sampled backgrounds."""

    def __init__(self, cfg: Optional[dict], *, cache_res: int):
        super().__init__()
        self.cfg = dict(cfg or {})
        self.prob = float(self.cfg.get("prob", 0.0))
        self.H = self.W = int(cache_res)
        alpha = self.cfg.get("alpha", [0.55, 0.75])
        if isinstance(alpha, (int, float)):
            alpha = [float(alpha), float(alpha)]
        self.alpha = (float(alpha[0]), float(alpha[1]))

        self.background_path = self.cfg.get("background_path")
        self.register_buffer("backgrounds_u8", None, persistent=False)
        if (
            self.prob > 0.0
            and self.background_path
            and os.path.exists(str(self.background_path))
        ):
            self.background_path = str(self.background_path)
            self.backgrounds_u8 = self._load_backgrounds()

    def forward(
        self,
        image: torch.Tensor,
        alpha=None,
        *,
        prob: Optional[float] = None,
        count: Optional[int] = None,
        replacement: Optional[bool] = None,
    ) -> torch.Tensor:
        if self.backgrounds_u8 is None:
            return image

        B = image.shape[0]
        if alpha is None:
            if not self.training or self.prob <= 0.0:
                return image
            replacement = True if replacement is None else bool(replacement)
            lo, hi = self.alpha
            alpha = lo + (hi - lo) * torch.rand(
                B, device=image.device, dtype=image.dtype
            )
            alpha = alpha.view(B, 1, 1, 1)
        else:
            replacement = False if replacement is None else bool(replacement)

        if torch.is_tensor(alpha):
            alpha = alpha.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
            if alpha.ndim == 1:
                alpha = alpha.view(B, 1, 1, 1)
            elif alpha.ndim == 3:
                alpha = alpha.unsqueeze(1)
            if alpha.ndim == 4 and alpha.shape[-2:] != image.shape[-2:]:
                alpha = F.interpolate(
                    alpha,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            alpha = max(0.0, min(1.0, float(alpha)))

        bg = self._sample_backgrounds(B, replacement=replacement).to(
            device=image.device,
            dtype=image.dtype,
        )
        if bg.shape[-2:] != image.shape[-2:]:
            bg = F.interpolate(
                bg,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        blended = image * alpha + bg * (1.0 - alpha)

        use_overlay = torch.zeros(B, 1, 1, 1, device=image.device, dtype=image.dtype)
        if count is None:
            apply_prob = self.prob if prob is None else float(prob)
            use_overlay = (torch.rand_like(use_overlay) < apply_prob).to(
                dtype=image.dtype
            )
        elif int(count) > 0:
            idx = torch.randperm(B, device=image.device)[: int(count)]
            use_overlay[idx] = 1.0
        return use_overlay * blended + (1.0 - use_overlay) * image

    def get_config(self) -> dict:
        return {
            "Status": (
                "Enabled"
                if self.backgrounds_u8 is not None and self.prob > 0
                else "Disabled"
            ),
            "Prob": self.prob,
            "Alpha": self.alpha,
            "Background": self.cfg.get("background_path"),
        }

    @torch.no_grad()
    def _sample_backgrounds(
        self,
        num_samples: int,
        *,
        replacement: bool = False,
    ) -> torch.Tensor:
        n = int(self.backgrounds_u8.shape[0])
        num_samples = int(num_samples)
        if replacement:
            idx = torch.randint(0, n, (num_samples,), device=self.backgrounds_u8.device)
        else:
            if num_samples > n:
                raise ValueError(
                    f"Requested {num_samples} backgrounds, but only {n} are loaded"
                )
            idx = torch.randperm(n, device=self.backgrounds_u8.device)[:num_samples]

        bg = self.backgrounds_u8.index_select(0, idx).to(torch.float32).div_(255.0)
        bg = self._random_transform(bg)
        return normalize_imagenet(bg, source="float01")

    def _random_transform(self, bg: torch.Tensor) -> torch.Tensor:
        if bg.shape[-2:] != (self.H, self.W):
            bg = F.interpolate(
                bg,
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            )

        B = bg.shape[0]
        angles = torch.rand(B, device=bg.device, dtype=bg.dtype) * 360.0
        bg = K.geometry.transform.rotate(
            bg,
            angles,
            mode="bilinear",
            padding_mode="border",
        )
        brightness = (torch.rand(B, device=bg.device, dtype=bg.dtype) * 0.2) - 0.1
        return K.enhance.adjust_brightness(bg, brightness).clamp(0.0, 1.0)

    def _load_backgrounds(self) -> torch.Tensor:
        if os.path.isfile(self.background_path):
            return self._load_background_pack(self.background_path)

        filenames = sorted(
            name
            for name in os.listdir(self.background_path)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not filenames:
            raise FileNotFoundError(
                f"No .jpg/.png images found in {self.background_path}"
            )

        out = torch.empty((len(filenames), 3, self.H, self.W), dtype=torch.uint8)
        for i, name in enumerate(filenames):
            path = os.path.join(self.background_path, name)
            image = Image.open(path).convert("RGB")
            if image.size != (self.W, self.H):
                image = image.resize((self.W, self.H), resample=Image.BILINEAR)
            arr = np.asarray(image, dtype=np.uint8)
            out[i] = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        return out

    def _load_background_pack(self, path: str) -> torch.Tensor:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if "backgrounds_u8" in payload:
                payload = payload["backgrounds_u8"]
            elif "images" in payload:
                payload = payload["images"]
            else:
                keys = ", ".join(sorted(str(key) for key in payload.keys()))
                raise KeyError(
                    f"Background pack {path} must contain 'backgrounds_u8' "
                    f"or 'images'; found keys: {keys}"
                )
        if not torch.is_tensor(payload):
            raise TypeError(f"Background pack {path} must contain a tensor")
        if payload.ndim != 4 or int(payload.shape[1]) != 3:
            raise ValueError(
                f"Background pack {path} must have shape [N,3,H,W], "
                f"got {tuple(payload.shape)}"
            )
        if payload.dtype != torch.uint8:
            payload = payload.clamp(0, 255).to(torch.uint8)
        if tuple(payload.shape[-2:]) != (self.H, self.W):
            payload = F.interpolate(
                payload.float(),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).round().clamp(0, 255).to(torch.uint8)
        return payload.contiguous()
