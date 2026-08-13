"""Remove frozen DINOv3 backbone weights from a checkpoint.

Two checkpoint shapes are supported:

- Seeker checkpoints (flat state_dict): drops all `vit.*` tensors.
  `Seeker.load_pretrained_weights` already loads the backbone separately
  (see `Seeker._init_dinov3`) and explicitly excludes `vit.*` keys from its
  strict-compatibility check.
- RVT2Heatmap checkpoints (payload dict with a `patch_backbone_state_dict`
  entry): drops that entire key when `patch_backbone == "dino"`.
  `RVT2Heatmap.__init__` (seeker/model/rvt2_heatmap.py) tolerates a missing
  `patch_backbone_state_dict` for a "dino" backbone, since
  `PatchFeatureBackbone` already loads the same frozen weights from
  `dino_ckpt_path` during construction.

In both cases the removed tensors are verified byte-identical to the
separately-distributed `dinov3.vits16plus.pth` before stripping is safe;
this script does not re-verify that per run, so only use it on checkpoints
whose DINOv3 backbone was actually frozen throughout training (never
fine-tuned).

Either way, this keeps the checkpoint's contents unambiguously
MIT-licensed Seeker code, independent of the separately-licensed
`dinov3.vits16plus.pth` (see seeker/model/dinov3_core/LICENSE.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def strip_seeker_backbone(state_dict: dict, *, prefix: str = "vit.") -> dict:
    kept = {k: v for k, v in state_dict.items() if not k.startswith(prefix)}
    removed = len(state_dict) - len(kept)
    print(f"[strip_dinov3_backbone] Removed {removed} '{prefix}*' tensors, kept {len(kept)}")
    return kept


def strip_rvt2_heatmap_backbone(payload: dict) -> dict:
    if "patch_backbone_state_dict" not in payload:
        print("[strip_dinov3_backbone] No patch_backbone_state_dict present; nothing to strip")
        return payload
    if str(payload.get("patch_backbone")) != "dino":
        raise ValueError(
            "Refusing to strip patch_backbone_state_dict: patch_backbone is "
            f"{payload.get('patch_backbone')!r}, not 'dino' (only the frozen "
            "DINOv3 backbone is safe to drop; a 'conv' backbone is trained)."
        )
    stripped = dict(payload)
    removed = len(stripped.pop("patch_backbone_state_dict"))
    print(f"[strip_dinov3_backbone] Removed patch_backbone_state_dict ({removed} tensors)")
    return stripped


def strip_backbone(payload) -> dict:
    if isinstance(payload, dict) and "patch_backbone_state_dict" in payload:
        return strip_rvt2_heatmap_backbone(payload)
    return strip_seeker_backbone(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the source checkpoint.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the backbone-stripped checkpoint to.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = Path(args.input)
    dst = Path(args.output)

    payload = torch.load(src, map_location="cpu")
    stripped = strip_backbone(payload)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, dst)
    print(f"[strip_dinov3_backbone] Saved: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
