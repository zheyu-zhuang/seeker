"""Image tensor normalization and resizing helpers."""

import torch
import torch.nn.functional as F


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SOURCE_MODES = {"raw", "uint8", "float01", "imagenet", "auto"}


def _stat_shape(x: torch.Tensor) -> list[int]:
    return [1] * max(x.ndim - 3, 0) + [3, 1, 1]


def _range(x: torch.Tensor) -> tuple[float, float]:
    if x.numel() == 0:
        return 0.0, 0.0
    return float(x.min().item()), float(x.max().item())


def _require_source(source: str) -> str:
    source = str(source)
    if source not in IMAGE_SOURCE_MODES:
        raise ValueError(f"Invalid image source {source!r}; expected {IMAGE_SOURCE_MODES}")
    return source


def _require_float01(x: torch.Tensor, *, source: str) -> torch.Tensor:
    if not x.is_floating_point():
        raise TypeError(f"Expected {source} image tensor to be floating point")
    min_v, max_v = _range(x)
    if min_v < 0.0 or max_v > 1.0:
        raise ValueError(
            f"Expected {source} image tensor in [0, 1], got range "
            f"[{min_v:.4g}, {max_v:.4g}]"
        )
    return x


def _uint_to_float01(x: torch.Tensor, *, source: str) -> torch.Tensor:
    if x.is_floating_point():
        raise TypeError(f"Expected {source} image tensor to be integer/uint8")
    min_v, max_v = _range(x)
    if min_v < 0.0 or max_v > 255.0:
        raise ValueError(
            f"Expected {source} image tensor in [0, 255], got range "
            f"[{min_v:.4g}, {max_v:.4g}]"
        )
    return x.float().div(255.0)


def _raw_to_float01(x: torch.Tensor) -> torch.Tensor:
    if not x.is_floating_point():
        return _uint_to_float01(x, source="raw")
    return _require_float01(x, source="raw")


def denorm_imagenet(x: torch.Tensor, *, force: bool = False) -> torch.Tensor:
    """Denormalize a tensor normalized with ImageNet stats."""
    min_v, _ = _range(x)
    if not force and min_v >= 0:
        return x
    stat_shape = _stat_shape(x)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(stat_shape)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(stat_shape)
    return x * std + mean


def image_to_float01(x: torch.Tensor, *, source: str = "raw") -> torch.Tensor:
    """Convert image tensors to float [0, 1] for visualization/augmentation."""
    source = _require_source(source)
    if source == "raw":
        x = _raw_to_float01(x)
    elif source == "uint8":
        x = _uint_to_float01(x, source=source)
    elif source == "float01":
        x = _require_float01(x, source=source)
    elif source == "imagenet":
        if not x.is_floating_point():
            raise TypeError("Expected imagenet image tensor to be floating point")
        x = denorm_imagenet(x, force=True)
    elif source == "auto":
        if not x.is_floating_point():
            x = x.float().div(255.0)
        else:
            min_v, max_v = _range(x)
            if min_v < 0.0 or (max_v > 1.0 and max_v <= 5.0):
                x = denorm_imagenet(x, force=True)
            elif max_v > 1.0:
                x = x.div(255.0)
    return x.clamp(0.0, 1.0)


def normalize_imagenet(x: torch.Tensor, *, source: str = "raw") -> torch.Tensor:
    """Normalize a tensor with ImageNet stats."""
    source = _require_source(source)
    if source == "raw":
        x = _raw_to_float01(x)
    elif source == "uint8":
        x = _uint_to_float01(x, source=source)
    elif source == "float01":
        x = _require_float01(x, source=source)
    elif source == "imagenet":
        if not x.is_floating_point():
            raise TypeError("Expected imagenet image tensor to be floating point")
        return x
    elif source == "auto":
        if not x.is_floating_point():
            x = x.float().div(255.0)
        else:
            min_v, max_v = _range(x)
            if min_v < 0.0 or (max_v > 1.0 and max_v <= 5.0):
                return x
            if max_v > 1.0:
                x = x.div(255.0)
    stat_shape = _stat_shape(x)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(stat_shape)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(stat_shape)
    return (x - mean) / std


def resize_image(image: torch.Tensor, out_res: int) -> torch.Tensor:
    """
    Resize image to out_res x out_res if needed.

    image: [N, 3, H, W]
    """
    _, _, H, W = image.shape
    if (H, W) != (out_res, out_res):
        image = F.interpolate(
            image,
            size=(out_res, out_res),
            mode="bilinear",
            align_corners=False,
        )
    return image
