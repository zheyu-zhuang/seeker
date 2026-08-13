"""Load and resolve repository paths from config and environment overrides."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
import os

from omegaconf import OmegaConf


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_PATHS_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "paths.yaml"


def _resolve_path(raw: str) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = _PACKAGE_ROOT / p
    return p.resolve()


@dataclass(frozen=True)
class PathConfig:
    """Resolved filesystem paths used across the project."""

    package_root: Path
    dataset_root: Path
    weights_dir: Path
    textures_dir: Path
    backgrounds_dir: Path
    task_emb_cache_path: Path


def load_path_config(
    overrides: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> PathConfig:
    """Load `config/paths.yaml` and return resolved absolute paths."""
    cfg_path = config_path or _PATHS_CFG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Path config not found: {cfg_path}")

    cfg_omega = OmegaConf.load(cfg_path)
    if overrides:
        cfg_omega = OmegaConf.merge(cfg_omega, OmegaConf.create(dict(overrides)))
    cfg = OmegaConf.to_container(cfg_omega, resolve=True)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Invalid path config format in {cfg_path}")

    paths = cfg.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("Missing/invalid 'paths' section in path config")

    required = (
        "dataset_root",
        "weights_dir",
        "textures_dir",
        "backgrounds_dir",
        "task_emb_cache",
    )
    missing = [k for k in required if k not in paths]
    if missing:
        raise ValueError(f"Missing path config keys: {missing}")

    dataset_root_raw = os.environ.get("ATTN_SEEKER_DATASET_ROOT", str(paths["dataset_root"]))
    weights_dir_raw = os.environ.get("ATTN_SEEKER_WEIGHTS_DIR", str(paths["weights_dir"]))
    textures_dir_raw = os.environ.get("ATTN_SEEKER_TEXTURES_DIR", str(paths["textures_dir"]))
    backgrounds_dir_raw = os.environ.get(
        "ATTN_SEEKER_BACKGROUNDS_DIR", str(paths["backgrounds_dir"])
    )
    task_emb_cache_raw = os.environ.get(
        "ATTN_SEEKER_TASK_CACHE_PATH", str(paths["task_emb_cache"])
    )

    dataset_root = _resolve_path(dataset_root_raw)
    weights_dir = _resolve_path(weights_dir_raw)
    textures_dir = _resolve_path(textures_dir_raw)
    backgrounds_dir = _resolve_path(backgrounds_dir_raw)
    task_emb_cache_path = _resolve_path(task_emb_cache_raw)

    return PathConfig(
        package_root=_PACKAGE_ROOT,
        dataset_root=dataset_root,
        weights_dir=weights_dir,
        textures_dir=textures_dir,
        backgrounds_dir=backgrounds_dir,
        task_emb_cache_path=task_emb_cache_path,
    )
