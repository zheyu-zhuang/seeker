"""Image byte encoding/decoding utilities used by dataset caches."""

from typing import Literal, Optional

import cv2
import numpy as np


ImageFormat = Literal["HWC", "CHW"]


def decode_jpg_bytes(
    buf: bytes,
    image_size: Optional[int] = None,
    *,
    bgr_to_rgb: bool = False,
    to_float: bool = True,
    fmt: ImageFormat = "CHW",
) -> np.ndarray:
    """
    Decode JPEG bytes -> image array.

    Args:
        buf: JPEG bytes
        image_size: if not None, resize to (image_size, image_size)
        bgr_to_rgb: set True to convert decoded OpenCV BGR output into RGB
        to_float: if True, returns float32 in [0, 1]; else uint8
        fmt: "HWC" or "CHW"

    Returns:
        np.ndarray:
          - if to_float: float32
          - else: uint8
          - layout: HWC or CHW
          - channels: BGR by default, RGB if bgr_to_rgb=True
    """
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # HWC uint8, BGR
    if img is None:
        raise ValueError("cv2.imdecode failed (buf may be corrupted)")

    if image_size is not None and (
        img.shape[0] != image_size or img.shape[1] != image_size
    ):
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if to_float:
        img = img.astype(np.float32) / 255.0

    if fmt == "CHW":
        img = np.moveaxis(img, -1, 0)

    return img


def encode_rgb_to_jpg_bytes(img_rgb: np.ndarray, quality: int = 90) -> bytes:
    """
    Expects:
        img_rgb: HWC uint8 image array
    Returns:
        JPEG bytes
    """
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)

    ok, buf = cv2.imencode(
        ".jpg", img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return buf.tobytes()
