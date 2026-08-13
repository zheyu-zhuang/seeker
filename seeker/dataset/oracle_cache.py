"""Frame-aligned MimicGen oracle labels for rerendered LMDB caches."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from seeker.dataset.cache import archive_path, load_metadata, load_numpy_array


@dataclass
class OracleContext:
    """MimicGen oracle helpers bound to the rerender environment."""

    task_spec: object = None
    interface: object = None
    affordance: Optional[object] = None

    @property
    def enabled(self) -> bool:
        return self.interface is not None

    @property
    def collect_oracle(self) -> bool:
        return self.task_spec is not None and self.affordance is not None


def load_oracle_info(cache_dir: str) -> Optional[dict[str, np.ndarray]]:
    """Load optional per-frame oracle arrays from a rerendered cache."""
    meta, _ = load_metadata(cache_dir)
    oracle_keys = [str(key) for key in meta.get("oracle_keys", [])]
    if oracle_keys:
        return {
            key: _normalize_loaded_oracle_array(
                load_numpy_array(cache_dir, f"oracle/{key}.npy")
            )
            for key in oracle_keys
        }

    packed_path = archive_path(cache_dir)
    info = {}
    with np.load(packed_path, allow_pickle=True) as archive:
        for key in sorted(archive.files):
            if not key.startswith("oracle/"):
                continue
            oracle_key = key[len("oracle/") :]
            if oracle_key.endswith(".npy"):
                oracle_key = oracle_key[:-4]
            info[oracle_key] = _normalize_loaded_oracle_array(np.asarray(archive[key]))
    return info if info else None


def _normalize_loaded_oracle_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind in ("U", "S"):
        return arr
    if arr.dtype == np.uint8 or arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if arr.dtype.kind in ("i", "u"):
        return arr.astype(np.int64)
    return arr.astype(np.float32)


def build_oracle_arrays(
    *,
    oracle_chunks: dict[str, list[np.ndarray]],
    expected_steps: int,
) -> dict[str, np.ndarray]:
    """Concatenate oracle chunks into archive-ready arrays."""
    arrays = {}
    for key in sorted(oracle_chunks.keys()):
        chunks = oracle_chunks[key]
        if not chunks:
            continue
        if key == "target_points":
            max_points = max(int(chunk.shape[1]) for chunk in chunks)
            padded_chunks = []
            for chunk in chunks:
                if int(chunk.shape[1]) < max_points:
                    pad = np.full(
                        (
                            chunk.shape[0],
                            max_points - int(chunk.shape[1]),
                            chunk.shape[2],
                        ),
                        np.nan,
                        dtype=chunk.dtype,
                    )
                    chunk = np.concatenate([chunk, pad], axis=1)
                padded_chunks.append(chunk)
            arr = np.concatenate(padded_chunks, axis=0)
        else:
            arr = np.concatenate(chunks, axis=0)
        if arr.shape[0] != expected_steps:
            raise ValueError(
                f"oracle/{key} has {arr.shape[0]} steps, expected {expected_steps}"
            )
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8)
        elif arr.dtype.kind not in ("U", "S", "i", "u"):
            arr = arr.astype(np.float32)
        arrays[key] = arr
    return arrays


def oracle_target_patch_mask_from_trajectory(
    *,
    traj: dict,
    camera: str,
) -> tuple[torch.Tensor, str]:
    """Load visible target masks on the rerender patch grid for one trajectory."""
    oracle_info = traj.get("oracle_info")
    if not oracle_info:
        raise RuntimeError(
            "Missing oracle_info. Rebuild the cache with `seeker rerender-dataset`."
        )

    mask_key = f"target_patch_mask_{camera}"
    if mask_key not in oracle_info:
        available = ", ".join(sorted(oracle_info.keys()))
        raise RuntimeError(
            f"Missing oracle/{mask_key}.npy in the rerendered cache. "
            "Rebuild it with `seeker rerender-dataset --oracle-camera "
            f"{camera}`. Available oracle keys: {available}"
        )

    masks = np.asarray(oracle_info[mask_key])
    if masks.ndim != 3:
        raise RuntimeError(
            f"Invalid oracle/{mask_key}.npy shape {masks.shape}; expected [T,S,S]."
        )
    mask_t = torch.from_numpy(masks.astype(np.uint8, copy=False)) > 0
    return mask_t, f"oracle.{mask_key}.visible_segmentation_patch_mask"


def oracle_transition_frames(traj: dict) -> list[int]:
    oracle_info = traj.get("oracle_info")
    if not oracle_info or "subtask_idx" not in oracle_info:
        return []
    subtask_idx = np.asarray(oracle_info["subtask_idx"]).reshape(-1)
    if subtask_idx.size < 2:
        return []
    valid = np.isfinite(subtask_idx.astype(np.float32, copy=False))
    transitions = []
    for idx in range(1, subtask_idx.size):
        if valid[idx] and valid[idx - 1] and subtask_idx[idx] != subtask_idx[idx - 1]:
            transitions.append(int(idx))
    return transitions


def build_oracle_context(
    *,
    env,
    env_meta: dict,
    camera_names: list[str],
    camera_name: str,
    resolution: int,
    patch_size: int,
) -> OracleContext:
    """Initialize MimicGen oracle helpers for a rerender environment."""
    from mimicgen.env_interfaces.base import make_interface

    from seeker.dataset.oracle_affordance import OracleAffordanceResolver
    from seeker.env.robomimic_image_wrapper import (
        _build_task_spec,
        _default_interface_name,
        _normalize_mimicgen_task_name,
    )

    if camera_name not in camera_names:
        print(
            f"[oracle] camera {camera_name!r} not in dataset cameras "
            f"{camera_names}; oracle geometry disabled"
        )
        return OracleContext()

    try:
        task_name = _normalize_mimicgen_task_name(env_meta["env_name"])
        task_spec = _build_task_spec(task_name)
        interface = make_interface(
            name=_default_interface_name(task_name),
            interface_type="robosuite",
            env=env.env,
        )
        affordance = OracleAffordanceResolver(
            env=env,
            env_meta=env_meta,
            camera_name=camera_name,
            resolution=resolution,
            patch_size=patch_size,
        )
        return OracleContext(
            task_spec=task_spec,
            interface=interface,
            affordance=affordance,
        )
    except Exception as exc:
        print(f"[oracle] disabled: could not initialize MimicGen oracle ({exc})")
        return OracleContext()


class OracleFrameCollector:
    """Callback that samples Seeker oracle targets."""

    @classmethod
    def build(
        cls,
        *,
        oracle: OracleContext,
        horizon: int,
        camera_name: str,
        resolution: int,
        patch_size: int,
        min_patch_area_fraction: float,
        min_mask_pixels: int = 16,
    ) -> Optional["OracleFrameCollector"]:
        """Return a collector only when oracle labels can be collected."""
        if not oracle.enabled or not oracle.collect_oracle:
            return None
        return cls(
            oracle=oracle,
            horizon=horizon,
            camera_name=camera_name,
            resolution=resolution,
            patch_size=patch_size,
            min_patch_area_fraction=min_patch_area_fraction,
            min_mask_pixels=min_mask_pixels,
        )

    def __init__(
        self,
        *,
        oracle: OracleContext,
        horizon: int,
        camera_name: str,
        resolution: int,
        patch_size: int,
        min_patch_area_fraction: float,
        min_mask_pixels: int = 16,
    ) -> None:
        self.oracle = oracle
        self.horizon = int(horizon)
        self.camera_name = str(camera_name)
        self.resolution = int(resolution)
        self.patch_size = int(patch_size)
        self.min_patch_area_fraction = float(min_patch_area_fraction)
        self.min_mask_pixels = int(min_mask_pixels)

        self.camera_static = self._camera_is_static(self.camera_name)
        self.static_camera_matrix = None
        self.active_subtask_idx = 0
        self.prev_signals = {}

        self.subtask_idx = np.zeros((self.horizon,), dtype=np.int64)
        self.subtask_signal = []
        self.object_ref = []
        self.object_xyz = np.full((self.horizon, 3), np.nan, dtype=np.float32)
        self.target_xyz = np.full((self.horizon, 3), np.nan, dtype=np.float32)
        self.target_points_per_frame = []
        self.target_box = np.full((self.horizon, 4), np.nan, dtype=np.float32)
        self.target_mask_area = np.full((self.horizon,), np.nan, dtype=np.float32)
        grid_size = int(np.ceil(float(self.resolution) / float(self.patch_size)))
        self.target_patch_mask = np.zeros(
            (self.horizon, grid_size, grid_size),
            dtype=np.uint8,
        )
        self.camera_matrix = np.full((self.horizon, 4, 4), np.nan, dtype=np.float32)

    def __call__(self, *, env, t: int, obs, state_dict) -> None:
        _ = obs, state_dict
        self._collect_oracle_info(env=env, t=int(t))

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Return episode-level oracle arrays."""
        camera_matrix_key = f"camera_matrix_{self.camera_name}"
        target_box_key = f"target_box_{self.camera_name}"
        target_patch_mask_key = f"target_patch_mask_{self.camera_name}"
        target_mask_area_key = f"target_mask_area_{self.camera_name}"
        return {
            "subtask_idx": self.subtask_idx,
            "subtask_term_signal": np.asarray(self.subtask_signal, dtype="<U128"),
            "object_ref": np.asarray(self.object_ref, dtype="<U128"),
            "object_xyz": self.object_xyz,
            "target_xyz": self.target_xyz,
            "target_points": self._padded_target_points(),
            target_box_key: self.target_box,
            target_patch_mask_key: self.target_patch_mask,
            target_mask_area_key: self.target_mask_area,
            camera_matrix_key: self.camera_matrix,
        }

    def _collect_oracle_info(self, *, env, t: int) -> None:
        datagen_info = self.oracle.interface.get_datagen_info(action=None)
        self.active_subtask_idx = self._advance_subtask(
            current_idx=self.active_subtask_idx,
            prev_signals=self.prev_signals,
            subtask_term_signals=datagen_info.subtask_term_signals,
        )
        self.prev_signals = self._scalar_signal_dict(datagen_info.subtask_term_signals)

        idx = int(self.active_subtask_idx)
        subtask = self.oracle.task_spec[idx]
        ref = subtask["object_ref"]

        self.subtask_idx[t] = idx
        self.subtask_signal.append(subtask["subtask_term_signal"] or "")
        self.object_ref.append(ref or "")
        if ref is None:
            self.target_points_per_frame.append(
                np.full((0, 3), np.nan, dtype=np.float32)
            )
            return

        pose = datagen_info.object_poses[ref]
        xyz = np.asarray(pose[:3, 3], dtype=np.float32)
        self.object_xyz[t] = xyz

        spec = self.oracle.affordance.affordance_spec(ref=ref, subtask_idx=idx)
        affordance_points = self.oracle.affordance.affordance_points(
            ref=ref,
            subtask_idx=idx,
            object_xyz=xyz,
            spec=spec,
        )
        affordance_points = np.asarray(affordance_points, dtype=np.float32).reshape(
            -1, 3
        )
        self.target_xyz[t] = np.mean(affordance_points, axis=0).astype(np.float32)
        self.target_points_per_frame.append(affordance_points)

        target_box, target_area, target_mask = self.oracle.affordance.segmentation_box(
            ref=ref,
            spec=spec,
            min_patch_area_fraction=self.min_patch_area_fraction,
            min_mask_pixels=self.min_mask_pixels,
        )
        if target_box is not None:
            self.target_box[t] = target_box
            self.target_mask_area[t] = float(target_area)
        if target_mask is not None:
            self.target_patch_mask[t] = np.asarray(target_mask, dtype=np.uint8)

        self.camera_matrix[t] = self._camera_transform_matrix(env)

    def _camera_transform_matrix(self, env) -> np.ndarray:
        from robosuite.utils.camera_utils import get_camera_transform_matrix

        if self.camera_static and self.static_camera_matrix is not None:
            return self.static_camera_matrix

        matrix = get_camera_transform_matrix(
            sim=env.env.sim,
            camera_name=self.camera_name,
            camera_height=self.resolution,
            camera_width=self.resolution,
        ).astype(np.float32)
        if self.camera_static:
            self.static_camera_matrix = matrix
        return matrix

    def _advance_subtask(self, current_idx, prev_signals, subtask_term_signals) -> int:
        idx = int(current_idx)
        task_spec = self.oracle.task_spec
        while idx < len(task_spec) - 1:
            signal = task_spec[idx]["subtask_term_signal"]
            if signal is None:
                break
            prev = int(prev_signals.get(signal, 0))
            cur = int(
                np.asarray(subtask_term_signals.get(signal, 0)).reshape(-1)[0]
            )
            if prev == 0 and cur == 1:
                idx += 1
                continue
            break
        return idx

    @staticmethod
    def _scalar_signal_dict(subtask_term_signals) -> dict:
        return {
            key: int(np.asarray(value).reshape(-1)[0])
            for key, value in subtask_term_signals.items()
        }

    def _padded_target_points(self) -> np.ndarray:
        max_points = max(
            (points.shape[0] for points in self.target_points_per_frame),
            default=0,
        )
        target_points = np.full(
            (len(self.target_points_per_frame), max(max_points, 1), 3),
            np.nan,
            dtype=np.float32,
        )
        for t, points in enumerate(self.target_points_per_frame):
            if points.size:
                target_points[t, : points.shape[0]] = points
        return target_points

    @staticmethod
    def _camera_is_static(camera_name: str) -> bool:
        camera = str(camera_name).lower()
        return "eye_in_hand" not in camera and not camera.startswith("robot")
