"""Action/cache helpers for MimicGen datasets."""

import os
from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

from seeker.dataset.cache import load_numpy_array
from seeker.util.formatting import is_matched_key


def validate_action_rep(action_rep):
    """Normalize and validate a Seeker action representation."""
    action_rep = str(action_rep)
    if action_rep in ("absolute", "delta"):
        return action_rep

    raise ValueError(
        f"Unsupported action_rep: {action_rep!r}. Supported values are: "
        "'absolute', 'delta'."
    )


def action_to_posmat(action: np.ndarray) -> np.ndarray:
    """Convert ``[T, 6+g]`` rotvec actions to ``[T, 3+9+g]`` pos+rotmat layout."""
    if action.ndim != 2 or action.shape[1] < 6:
        raise ValueError(f"action_to_posmat expects (T, >=6), got {action.shape}")
    pos = action[:, :3]
    rotvec = action[:, 3:6]
    gripper = action[:, 6:]
    rot_mat = R.from_rotvec(rotvec).as_matrix()
    rot_9d = rot_mat.reshape(rot_mat.shape[0], 9)
    return np.concatenate([pos, rot_9d, gripper], axis=1)


def action_posmat_to_pos6d(action: np.ndarray) -> np.ndarray:
    """Convert pos+rotmat+gripper action to pos+rot6d+gripper layout."""
    pos = action[:, :3]
    rot = action[:, 3:12]
    gripper = action[:, 12:]
    rot_6d = rot[:, :6]
    return np.concatenate([pos, rot_6d, gripper], axis=1)


def absolute_posmat_to_delta_chunks(
    eef_pos: np.ndarray,
    eef_rot: np.ndarray,
    action_posmat: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Build first-frame-relative chunked delta actions from absolute pos+rotmat actions."""
    horizon = int(horizon)
    num_steps, action_dim = action_posmat.shape
    if action_dim < 12:
        raise ValueError(
            f"absolute_action must have >=12 dims (3+9), got {action_dim}"
        )
    if eef_pos.shape[0] != num_steps or eef_rot.shape[0] != num_steps:
        raise ValueError("eef_pos/eef_rot length mismatch with action")

    out = []
    for i in range(action_posmat.shape[0]):
        chunk = action_posmat[i : i + horizon]
        if chunk.shape[0] < horizon:
            pad = horizon - chunk.shape[0]
            chunk = np.concatenate([chunk, np.tile(chunk[-1:], (pad, 1))], axis=0)

        r0 = eef_rot[i].reshape(3, 3)
        p0 = eef_pos[i].reshape(3)

        pt = chunk[:, :3]
        rt = chunk[:, 3:12].reshape(-1, 3, 3)
        gr = chunk[:, 12:]

        dp_world = pt - p0
        dp_body = (r0.T @ dp_world.T).T
        r_rel = r0.T[None, :, :] @ rt
        rot_delta = r_rel.reshape(-1, 9)[..., :6]

        delta = np.concatenate([dp_body, rot_delta, gr], axis=1)
        out.append(delta.reshape(1, -1))

    return np.concatenate(out, axis=0).astype(np.float32)


def load_action_array(
    *,
    cache_dir: str,
    meta: Dict,
    horizon: int,
    action_dim: int,
    action_rep: str,
) -> np.ndarray:
    """Load action array for the selected action representation."""
    action_rep = validate_action_rep(action_rep)
    if action_rep == "delta":
        return load_delta_action(
            cache_dir=cache_dir,
            meta=meta,
            horizon=horizon,
            action_dim=action_dim,
        ).astype(np.float32, copy=False)

    action = load_numpy_array(cache_dir, os.path.join("action", f"{action_rep}_action.npy"))
    action = action_posmat_to_pos6d(action)
    return action[:, :action_dim].astype(np.float32, copy=False)


def load_delta_action(
    *,
    cache_dir: str,
    meta: Dict,
    horizon: int,
    action_dim: int,
) -> np.ndarray:
    """Load precomputed chunked delta actions and project per-step dims."""
    rel_path = os.path.join("action", f"delta_action_h{int(horizon)}.npy")
    try:
        action = load_numpy_array(cache_dir, rel_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing precomputed delta-action cache "
            f"{rel_path!r}. Rebuild rerender/merged cache with "
            f"--delta-horizons {int(horizon)} or train with action_rep=absolute."
        ) from exc
    if action.ndim != 2:
        raise ValueError(f"delta-action cache must be 2D, got shape {action.shape}")
    if action.shape[1] % horizon != 0:
        raise ValueError(
            "delta-action cache width must be divisible by the requested "
            f"horizon, got width={action.shape[1]} horizon={horizon}"
        )

    per_step_dim = action.shape[1] // horizon
    if per_step_dim < action_dim:
        raise ValueError(
            "delta action per-step dim is smaller than configured action dim: "
            f"{per_step_dim} < {action_dim}"
        )

    action = action.reshape(-1, horizon, per_step_dim)
    return action[..., :action_dim].reshape(action.shape[0], -1)


def load_required_lowdim(
    *,
    cache_dir: str,
    meta: Dict,
    pattern: str,
) -> np.ndarray:
    """Load one required cached lowdim array matched by pattern."""
    matches = [key for key in meta.get("lowdim_keys", []) if is_matched_key(pattern, key)]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one lowdim key matching {pattern!r}, got {matches}"
        )
    key = matches[0]
    return load_numpy_array(cache_dir, os.path.join("lowdim", f"{key}.npy"))
