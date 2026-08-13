"""RVT2Heatmap heuristic labels, keypoints, and projection helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from seeker.dataset.mimicgen_dataset import MimicGenDataset
from seeker.focus_toolbox.config import load_rvt2_heatmap_config


def gripper_open_from_signal(gripper: np.ndarray) -> np.ndarray:
    """Convert continuous or binary gripper observations to open/closed booleans."""
    g = np.asarray(gripper)
    if g.ndim == 0:
        raise ValueError("gripper must contain one value per timestep")
    if g.ndim > 2:
        g = g.reshape(g.shape[0], -1)

    if g.ndim == 2 and g.shape[1] == 2:
        signal = np.abs(g[:, 0] - g[:, 1])
    elif g.ndim == 2:
        signal = g[:, 0]
    else:
        signal = g

    signal = np.asarray(signal).reshape(-1)
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        raise ValueError("gripper signal has no finite values")

    uniq = np.unique(finite)
    if uniq.size <= 2:
        return signal > float(np.mean(uniq))

    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if np.isclose(lo, hi):
        return np.ones(signal.shape[0], dtype=bool)
    return signal > (lo + hi) * 0.5


def discover_rvt2_heatmap_keypoints(
    joint_velocities: np.ndarray,
    gripper_open: Sequence[bool] | np.ndarray,
    *,
    atol: float = 0.1,
    stopped_buffer_len: int = 4,
    include_gripper_changes: bool = True,
    include_final: bool = True,
    mute_initial_gripper_open: bool = False,
    remove_adjacent: bool = True,
) -> Tuple[np.ndarray, Dict[int, List[str]]]:
    """
    Discover keypoints with the fixed heuristic used by the RVT2Heatmap baseline.

    A timestep is selected when the gripper state changes, the robot has stopped
    while the gripper state is stable around that frame, or it is the final frame.
    """
    velocities = np.asarray(joint_velocities, dtype=np.float32)
    if velocities.ndim == 1:
        velocities = velocities[:, None]
    if velocities.ndim != 2:
        raise ValueError(f"joint_velocities must be [T,D], got {velocities.shape}")

    gripper = np.asarray(gripper_open)
    if gripper.dtype != np.bool_:
        gripper = gripper_open_from_signal(gripper)
    else:
        gripper = gripper.reshape(-1)

    T = int(velocities.shape[0])
    if gripper.shape[0] != T:
        raise ValueError(
            f"gripper_open length mismatch: got {gripper.shape[0]}, expected {T}"
        )
    if T == 0:
        return np.empty((0,), dtype=int), {}

    keypoints: List[int] = []
    reasons: Dict[int, List[str]] = {}
    prev_gripper_open = bool(gripper[0])
    stopped_buffer = 0

    for i in range(T):
        gripper_stable = (
            i < T - 2
            and bool(gripper[i]) == bool(gripper[i + 1])
            and bool(gripper[i]) == bool(gripper[max(0, i - 1)])
            and bool(gripper[max(0, i - 2)]) == bool(gripper[max(0, i - 1)])
        )
        stopped = (
            stopped_buffer <= 0
            and np.allclose(velocities[i], 0.0, atol=float(atol))
            and gripper_stable
            and i != T - 2
        )

        if stopped:
            stopped_buffer = int(stopped_buffer_len)
        else:
            stopped_buffer -= 1

        last = include_final and i == T - 1
        changed = bool(gripper[i]) != prev_gripper_open

        if i != 0 and ((include_gripper_changes and changed) or stopped or last):
            keypoints.append(i)
            frame_reasons = []
            if include_gripper_changes and changed:
                frame_reasons.append("gripper")
            if stopped:
                frame_reasons.append("stopped")
            if last:
                frame_reasons.append("final")
            reasons[i] = frame_reasons

        prev_gripper_open = bool(gripper[i])

    if remove_adjacent and len(keypoints) >= 2 and keypoints[-1] == keypoints[-2] + 1:
        removed = keypoints.pop(-2)
        reasons.pop(removed, None)

    if mute_initial_gripper_open:
        for keypoint in list(keypoints):
            frame_reasons = reasons.get(int(keypoint), [])
            if frame_reasons == ["gripper"] and bool(gripper[int(keypoint)]):
                keypoints.remove(keypoint)
                reasons.pop(int(keypoint), None)
                break

    return np.asarray(keypoints, dtype=int), reasons


RVT2_HEATMAP_CONFIG = load_rvt2_heatmap_config()
JOINT_VELOCITY_KEY = "robot0_joint_vel"
RVT2_HEATMAP_JOINT_VEL_ATOL = RVT2_HEATMAP_CONFIG["joint_vel_atol"]
RVT2_HEATMAP_STOPPED_BUFFER_LEN = RVT2_HEATMAP_CONFIG["stopped_buffer_len"]
RVT2_HEATMAP_INCLUDE_FINAL = RVT2_HEATMAP_CONFIG["include_final"]
RVT2_HEATMAP_MUTE_INITIAL_GRIPPER_OPEN = (
    RVT2_HEATMAP_CONFIG["mute_initial_gripper_open"]
)
RVT2_HEATMAP_KEYPOINT_BOX_ZOOM = RVT2_HEATMAP_CONFIG["keypoint_box_zoom"]


def _box_from_row_col(row_col: np.ndarray, image_size: int, zoom: float) -> np.ndarray:
    row = float(row_col[0])
    col = float(row_col[1])
    side = max(2.0, float(image_size) / max(float(zoom), 1e-6))
    x1 = np.clip(col - side / 2.0, 0.0, image_size - 1.0)
    y1 = np.clip(row - side / 2.0, 0.0, image_size - 1.0)
    x2 = np.clip(col + side / 2.0, 0.0, image_size - 1.0)
    y2 = np.clip(row + side / 2.0, 0.0, image_size - 1.0)
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)

def _project_world_xyz_to_row_col(
    xyz: np.ndarray,
    world_to_pixel: np.ndarray,
    image_size: int,
) -> np.ndarray | None:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(3)
    matrix = np.asarray(world_to_pixel, dtype=np.float32)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    if not np.isfinite(xyz).all():
        return None

    hom = np.asarray([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float32)
    projected = matrix @ hom
    depth = float(projected[2])
    if not np.isfinite(depth) or abs(depth) <= 1e-8:
        return None

    col = float(projected[0] / depth)
    row = float(projected[1] / depth)
    if not np.isfinite(row) or not np.isfinite(col):
        return None
    row = float(np.clip(round(row), 0, image_size - 1))
    col = float(np.clip(round(col), 0, image_size - 1))
    return np.asarray([row, col], dtype=np.float32)

def rvt2_heatmap_keypoints_for_trajectory(
    traj: dict,
    dataset: MimicGenDataset,
    episode_index: int,
) -> tuple[np.ndarray, dict[int, list[str]], str]:
    """Compute RVT2Heatmap heuristic keypoints for a cached trajectory."""
    obs = traj["obs"]

    joint_velocities, velocity_source = dataset.optional_episode_lowdim(
        episode_index, JOINT_VELOCITY_KEY
    )
    if joint_velocities is None:
        raise ValueError(
            "RVT2Heatmap heuristic keypoints require cached joint velocities; "
            f"episode {episode_index} is missing lowdim/{JOINT_VELOCITY_KEY}.npy."
        )

    if "robot0_gripper_qpos" in obs:
        gripper_open = gripper_open_from_signal(obs["robot0_gripper_qpos"])
        gripper_source = "robot0_gripper_qpos"
    else:
        gripper_open = gripper_open_from_signal(traj["action"][:, -1])
        gripper_source = "action[-1]"

    keypoints, reasons = discover_rvt2_heatmap_keypoints(
        joint_velocities,
        gripper_open,
        atol=RVT2_HEATMAP_JOINT_VEL_ATOL,
        stopped_buffer_len=RVT2_HEATMAP_STOPPED_BUFFER_LEN,
        include_gripper_changes=True,
        include_final=RVT2_HEATMAP_INCLUDE_FINAL,
        mute_initial_gripper_open=RVT2_HEATMAP_MUTE_INITIAL_GRIPPER_OPEN,
    )
    events = "stopped_gripper_changes"
    if RVT2_HEATMAP_INCLUDE_FINAL:
        events = f"{events}_and_final"
    if RVT2_HEATMAP_MUTE_INITIAL_GRIPPER_OPEN:
        events = f"{events}_mute_initial_gripper_open"
    source = (
        f"velocity={velocity_source}, gripper={gripper_source}, "
        f"atol={RVT2_HEATMAP_JOINT_VEL_ATOL:g}, "
        f"events={events}"
    )
    return keypoints, reasons, source

def format_keypoint_labels(reasons: dict[int, list[str]]) -> dict[int, str]:
    return {
        int(i): f"RVT2Heatmap: {', '.join(reason) if reason else 'keypoint'}"
        for i, reason in reasons.items()
    }

def eef_xyz_for_trajectory(traj: dict) -> tuple[np.ndarray | None, str]:
    """Return actual dataset EEF XYZ for RVT2Heatmap heuristic keyframe poses."""
    obs = traj["obs"]
    if "robot0_eef_pos" not in obs:
        return None, "missing_robot0_eef_pos"
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float32), "robot0_eef_pos"

def next_keypoint_projected_boxes(
    *,
    keypoints: np.ndarray,
    xyz: np.ndarray | None,
    pose_source: str,
    camera_matrices: np.ndarray | None,
    image_size: int,
) -> tuple[torch.Tensor | None, str]:
    """Project XYZ at the next keyframe using cached camera matrices."""
    if xyz is None:
        return None, pose_source
    if camera_matrices is None:
        return None, "missing_camera_matrix"

    T = int(xyz.shape[0])
    camera_matrices = np.asarray(camera_matrices, dtype=np.float32)
    if camera_matrices.shape[:1] != (T,) or camera_matrices.shape[1:] != (4, 4):
        return None, f"invalid_camera_matrix_shape_{camera_matrices.shape}"

    idx = np.array(sorted(set(int(i) for i in keypoints)), dtype=int)
    idx = idx[(idx >= 0) & (idx < T)]
    if idx.size == 0:
        return torch.full((T, 4), float("nan"), dtype=torch.float32), pose_source

    projected = {}
    for target_idx in idx.tolist():
        row_col = _project_world_xyz_to_row_col(
            xyz[int(target_idx)],
            camera_matrices[int(target_idx)],
            image_size=image_size,
        )
        if row_col is None:
            continue
        projected[int(target_idx)] = _box_from_row_col(
            row_col,
            image_size=image_size,
            zoom=RVT2_HEATMAP_KEYPOINT_BOX_ZOOM,
        )

    out_np = np.full((T, 4), np.nan, dtype=np.float32)
    keypoint_pos = 0
    for frame_idx in range(T):
        while keypoint_pos < idx.size and idx[keypoint_pos] <= frame_idx:
            keypoint_pos += 1
        if keypoint_pos >= idx.size:
            continue
        target_idx = int(idx[keypoint_pos])
        if target_idx in projected:
            out_np[frame_idx] = projected[target_idx]
    return torch.from_numpy(out_np).float(), pose_source
