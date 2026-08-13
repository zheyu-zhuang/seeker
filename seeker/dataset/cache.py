"""Cache helpers for MimicGen datasets."""

from __future__ import annotations

import json
import os
import fcntl
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ARRAYS_ARCHIVE_NAME = "arrays.npz"
ARRAYS_UNPACK_DIR_NAME = ".arrays_npy"
ARRAYS_UNPACK_DONE_NAME = ".arrays_npy_done"
ARRAYS_UNPACK_LOCK_NAME = ".arrays_npy.lock"


def archive_path(cache_dir: str | os.PathLike) -> Path:
    """Return the packed NumPy archive path for a cache directory."""
    return Path(cache_dir).expanduser().resolve() / ARRAYS_ARCHIVE_NAME


def archive_key(rel_path: str | os.PathLike) -> str:
    """Normalize a cache-relative ``.npy`` path to an ``arrays.npz`` key."""
    key = str(rel_path).replace(os.sep, "/")
    if key.endswith(".npy"):
        key = key[: -len(".npy")]
    return key


def unpacked_arrays_dir(cache_dir: str | os.PathLike) -> Path:
    """Return the derived mmap-friendly unpack directory for packed arrays."""
    return Path(cache_dir).expanduser().resolve() / ARRAYS_UNPACK_DIR_NAME


def _unpacked_array_path(cache_dir: str | os.PathLike, key: str) -> Path:
    return unpacked_arrays_dir(cache_dir) / f"{key}.npy"


def _unpack_is_current(cache_dir: str | os.PathLike) -> bool:
    packed_path = archive_path(cache_dir)
    done_path = unpacked_arrays_dir(cache_dir) / ARRAYS_UNPACK_DONE_NAME
    return done_path.exists() and done_path.stat().st_mtime >= packed_path.stat().st_mtime


def ensure_unpacked_arrays(cache_dir: str | os.PathLike) -> Path:
    """Materialize packed arrays as local .npy files for mmap-based training."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    packed_path = archive_path(cache_dir)
    if not packed_path.exists():
        raise FileNotFoundError(f"Missing packed array archive: {packed_path}")

    unpack_dir = unpacked_arrays_dir(cache_dir)
    if _unpack_is_current(cache_dir):
        return unpack_dir

    lock_path = cache_dir / ARRAYS_UNPACK_LOCK_NAME
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if _unpack_is_current(cache_dir):
            return unpack_dir
        if unpack_dir.exists():
            shutil.rmtree(unpack_dir)
        unpack_dir.mkdir(parents=True)
        with zipfile.ZipFile(packed_path, "r") as archive:
            members = [name for name in archive.namelist() if name.endswith(".npy")]
            for member in members:
                out_path = unpack_dir / member
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        done_path = unpack_dir / ARRAYS_UNPACK_DONE_NAME
        done_path.write_text("arrays unpacked\n")
    return unpack_dir


def is_cache_dir(path: str | os.PathLike) -> bool:
    """Return whether path looks like a rerendered cache directory."""
    path = Path(path)
    return (
        path.is_dir()
        and (path / "meta.json").exists()
        and (path / "images.lmdb").exists()
        and (path / ARRAYS_ARCHIVE_NAME).exists()
        and (path / "build_done.flag").exists()
    )


def default_cache_dir_for_dataset(
    dataset_path: str | os.PathLike,
    output_root: str | os.PathLike | None = None,
) -> Path:
    dataset_path = Path(dataset_path).expanduser().resolve()
    task_name = dataset_path.stem
    if output_root is None:
        task_dir = dataset_path.parent
    else:
        task_dir = Path(output_root).expanduser().resolve() / task_name
    return task_dir / f"{task_name}_lmdb"


def resolve_cache_dir(dataset_path: str, cache_dir: str | None = None) -> Path:
    """Resolve cache dir from an explicit cache, cache dir, task dir, or raw HDF5."""
    path = Path(dataset_path).expanduser().resolve()
    if cache_dir is not None:
        resolved = Path(cache_dir).expanduser().resolve()
    elif is_cache_dir(path):
        resolved = path
    elif path.is_dir():
        resolved = path / f"{path.name}_lmdb"
    else:
        resolved = default_cache_dir_for_dataset(path)
    if not is_cache_dir(resolved):
        raise FileNotFoundError(
            "Expected a rerendered cache directory with meta.json, images.lmdb, "
            f"{ARRAYS_ARCHIVE_NAME}, and build_done.flag: "
            f"{resolved}"
        )
    return resolved


def get_obs_keys(shape_meta: Dict) -> Tuple[List[str], List[str]]:
    """Split observation keys from ``shape_meta`` into RGB and lowdim groups."""
    rgb_keys, lowdim_keys = [], []
    for key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            rgb_keys.append(key)
        elif obs_type == "low_dim":
            lowdim_keys.append(key)
    return rgb_keys, lowdim_keys


def load_numpy_array(cache_dir: str, rel_path: str) -> np.ndarray:
    """Load a cache array, using mmap-friendly unpacked ``.npy`` files."""
    key = archive_key(rel_path)
    packed_path = archive_path(cache_dir)
    if not packed_path.exists():
        raise FileNotFoundError(f"Missing packed array archive: {packed_path}")
    ensure_unpacked_arrays(cache_dir)
    path = _unpacked_array_path(cache_dir, key)
    if not path.exists():
        raise FileNotFoundError(f"Missing array {key!r} in {packed_path}")
    try:
        return np.load(path, mmap_mode="r", allow_pickle=True)
    except ValueError:
        return np.load(path, allow_pickle=True)


def load_metadata(cache_dir: str) -> Tuple[Dict, List[int]]:
    """Load cache metadata and episode length table from ``meta.json``."""
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta, list(map(int, meta["episode_lengths"]))


def load_lowdim(cache_dir: str, lowdim_keys: List[str]) -> Dict[str, np.ndarray]:
    """Load all low-dimensional observation arrays declared in ``shape_meta``."""
    return {
        key: load_numpy_array(cache_dir, os.path.join("lowdim", f"{key}.npy")).astype(
            np.float32,
            copy=False,
        )
        for key in lowdim_keys
    }


def load_task_instructions(cache_dir: str) -> List[str] | None:
    """Load optional per-episode task instructions as Python strings."""
    try:
        arr = load_numpy_array(cache_dir, "task_instructions.npy")
    except FileNotFoundError:
        return None
    return [
        inst.decode("utf-8") if isinstance(inst, (bytes, np.bytes_)) else str(inst)
        for inst in arr
    ]


def check_cache(cache_dir: str, dataset_path: str) -> None:
    """Validate cache directory existence and presence of core cache files."""
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(
            "[MimicGenDataset] Cache dir not found.\n"
            f"  dataset_path: {os.path.abspath(os.path.expanduser(dataset_path))}\n"
            f"  cache_dir   : {cache_dir}\n"
            "\nBuild the cache first, e.g.:\n"
            "  seeker rerender-dataset --dataset <raw_dataset.hdf5>\n"
        )

    missing = [
        os.path.join(cache_dir, name)
        for name in ("images.lmdb", "meta.json", ARRAYS_ARCHIVE_NAME, "build_done.flag")
        if not os.path.exists(os.path.join(cache_dir, name))
    ]
    if not missing:
        return

    raise FileNotFoundError(
        "[MimicGenDataset] Cache looks incomplete (missing core files).\n"
        f"  dataset_path: {os.path.abspath(os.path.expanduser(dataset_path))}\n"
        f"  cache_dir   : {os.path.abspath(cache_dir)}\n"
        "  missing:\n"
        + "\n".join([f"    - {path}" for path in missing])
        + "\n\nBuild the cache first, e.g.:\n"
        "  seeker rerender-dataset --dataset <raw_dataset.hdf5>\n"
    )
