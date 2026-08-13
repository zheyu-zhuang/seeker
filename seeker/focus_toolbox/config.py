"""Config loading for RVT2Heatmap defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from seeker import CONFIG_DIR, REPO_ROOT

_RVT2_CFG_PATH = CONFIG_DIR / "rvt2_default_params.yaml"


def _repo_path(path: str) -> str:
    path = Path(str(path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def load_rvt2_default_config_dict(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = OmegaConf.load(_RVT2_CFG_PATH)
    if overrides:
        cfg = OmegaConf.merge(cfg, dict(overrides))
    return dict(OmegaConf.to_container(cfg, resolve=True))


def load_rvt2_query_config(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_rvt2_default_config_dict(overrides)["query_composer"]
    return {
        "proprio_mode": str(cfg["proprio_mode"]),
        "task_emb_dim": int(cfg["task_emb_dim"]),
        "hidden_mult": int(cfg["hidden_mult"]),
        "proprio_dim": int(cfg["proprio_dim"]),
    }


def load_rvt2_heatmap_config(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_rvt2_default_config_dict(overrides)["rvt2_heatmap"]
    return {
        "camera": str(cfg["camera"]),
        "patch_backbone": str(cfg["patch_backbone"]).lower(),
        "conv_patch_dim": int(cfg["conv_patch_dim"]),
        "dino_ckpt": _repo_path(cfg["dino_ckpt"]),
        "dino_image_size": int(cfg["dino_image_size"]),
        "patch_size": int(cfg["patch_size"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "transformer_depth": int(cfg["transformer_depth"]),
        "transformer_heads": int(cfg["transformer_heads"]),
        "transformer_dropout": float(cfg["transformer_dropout"]),
        "target_sigma_patches": float(cfg["target_sigma_patches"]),
        "joint_vel_atol": float(cfg["joint_vel_atol"]),
        "stopped_buffer_len": int(cfg["stopped_buffer_len"]),
        "include_final": bool(cfg["include_final"]),
        "mute_initial_gripper_open": bool(cfg["mute_initial_gripper_open"]),
        "keypoint_box_zoom": float(cfg["keypoint_box_zoom"]),
    }
