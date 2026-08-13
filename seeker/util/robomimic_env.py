"""Robomimic/robosuite integration helpers for actions, textures, and rollouts."""

from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import os
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.envs.env_base import EnvBase
from scipy.spatial.transform import Rotation as R
from seeker.util.mjcf_texture import apply_table_texture

import cv2

logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)

CONTROLLER_MODES = ("absolute", "relative")
ACTION_REPRESENTATIONS = (*CONTROLLER_MODES, "delta")
CONTROLLER_MODE_BY_ACTION_REP = {
    "absolute": "absolute",
    "relative": "relative",
    "delta": "absolute",
}
_TABLE_TEXTURE_FILE: ContextVar[Optional[str]] = ContextVar(
    "_TABLE_TEXTURE_FILE", default=None
)
_HOOK_INSTALLED = False


# ------------------------------ Action Utilities ------------------------------ #

def camera_pose_from_xml(pos, quat):
    """Convert XML camera pose (pos, wxyz quat) to world transform matrix."""
    X_C = np.eye(4)
    obj_quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    X_C[:3, :3] = R.from_quat(obj_quat_xyzw).as_matrix()
    X_C[:3, 3] = pos
    camera_axis_correction = np.diag([1, -1, -1, 1])
    return X_C @ camera_axis_correction


def validate_action_rep(action_rep):
    """Normalize and validate a Seeker action representation."""
    action_rep = str(action_rep)
    if action_rep in ACTION_REPRESENTATIONS:
        return action_rep

    supported = ", ".join(repr(x) for x in ACTION_REPRESENTATIONS)
    raise ValueError(
        f"Unsupported action_rep: {action_rep!r}. Supported values are: {supported}."
    )


def action_to_posmat(action: np.ndarray) -> np.ndarray:
    """Convert `[T, 6+g]` (pos+rotvec+grip) to `[T, 3+9+g]` pos+rotmat format."""
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
    rot_repr: str = "6d",
) -> np.ndarray:
    """Build first-frame-relative chunked delta actions from absolute pos+rotmat actions.

    Each chunk is anchored to the end-effector pose at its first frame. This is not
    a pairwise timestep-to-timestep finite difference encoding.
    """
    horizon = int(horizon)
    rot_repr = str(rot_repr)
    _, action_dim = action_posmat.shape
    if action_dim < 12:
        raise ValueError(
            f"absolute_action must have >=12 dims (3+9), got {action_dim}"
        )
    num_steps = action_posmat.shape[0]
    if eef_pos.shape[0] != num_steps or eef_rot.shape[0] != num_steps:
        raise ValueError("eef_pos/eef_rot length mismatch with action")
    if rot_repr not in ("6d", "mat"):
        raise ValueError(
            f"Unsupported rot_repr: {rot_repr!r}. Supported values are '6d' and 'mat'."
        )

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
        if rot_repr == "6d":
            rot_delta = r_rel.reshape(-1, 9)[..., :6]
        else:
            rot_delta = r_rel.reshape(-1, 9)

        delta = np.concatenate([dp_body, rot_delta, gr], axis=1)
        out.append(delta.reshape(1, -1))

    return np.concatenate(out, axis=0).astype(np.float32)


def update_env_controller(env_meta, action_rep):
    """Patch controller config according to the requested action representation."""
    action_rep = validate_action_rep(action_rep)
    controller_mode = CONTROLLER_MODE_BY_ACTION_REP[action_rep]

    env_meta = deepcopy(env_meta)
    env_meta["env_kwargs"]["controller_configs"]["action_mode"] = controller_mode
    return env_meta


def get_action_key(controller_mode, original_key="actions"):
    """Return HDF5 key name for controller-backed action modes."""
    if controller_mode == "all":
        return [f"{mode}_{original_key}" for mode in CONTROLLER_MODES] + [
            original_key
        ]
    if controller_mode == "default":
        return original_key
    if controller_mode not in CONTROLLER_MODES:
        supported = ", ".join(repr(x) for x in CONTROLLER_MODES)
        raise ValueError(
            f"Unsupported controller_mode: {controller_mode!r}. Supported values are: {supported}."
        )
    return f"{controller_mode}_{original_key}"


# ------------------------------ Texture Utilities ----------------------------- #

def set_table_texture_file(texture_file: Optional[str]) -> None:
    """Set process-local table texture path used by XML model hooks."""
    _TABLE_TEXTURE_FILE.set(texture_file)


@contextmanager
def table_texture(texture_file: Optional[str]):
    """Context manager that temporarily overrides table texture file."""
    token = _TABLE_TEXTURE_FILE.set(texture_file)
    try:
        yield
    finally:
        _TABLE_TEXTURE_FILE.reset(token)


def install_table_texture_hook() -> None:
    """Patch MuJoCo model constructors to inject table textures."""
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return

    import mujoco

    orig_from_xml_string = mujoco.MjModel.from_xml_string

    def patched_from_xml_string(xml: str, assets=None):
        tex = _TABLE_TEXTURE_FILE.get()
        if tex is not None:
            try:
                xml = apply_table_texture(xml, texture_file=tex)
            except Exception:
                pass
        return orig_from_xml_string(xml, assets=assets)

    mujoco.MjModel.from_xml_string = patched_from_xml_string

    orig_from_xml_path = getattr(mujoco.MjModel, "from_xml_path", None)
    if orig_from_xml_path is not None:

        def patched_from_xml_path(path: str, assets=None):
            tex = _TABLE_TEXTURE_FILE.get()
            if tex is None:
                return orig_from_xml_path(path, assets=assets)

            tmp_path = None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    xml = f.read()
                xml = apply_table_texture(xml, texture_file=tex)

                # Keep path-context by writing patched XML next to source XML.
                xml_dir = os.path.dirname(os.path.abspath(path))
                fd, tmp_path = tempfile.mkstemp(
                    prefix=".table_tex_", suffix=".xml", dir=xml_dir, text=True
                )
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(xml)
                return orig_from_xml_path(tmp_path, assets=assets)
            except Exception:
                # Fallback to original loader to avoid breaking env creation.
                return orig_from_xml_path(path, assets=assets)
            finally:
                if tmp_path is not None:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        mujoco.MjModel.from_xml_path = patched_from_xml_path

    try:
        import mujoco_py

        orig_load = mujoco_py.load_model_from_xml

        def patched_load_model_from_xml(xml: str):
            tex = _TABLE_TEXTURE_FILE.get()
            if tex is not None:
                try:
                    xml = apply_table_texture(xml, texture_file=tex)
                except Exception:
                    pass
            return orig_load(xml)

        mujoco_py.load_model_from_xml = patched_load_model_from_xml
    except Exception:
        pass

    _HOOK_INSTALLED = True


# ------------------------------ Rollout Utilities ----------------------------- #

def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    reset_every_step=False,
    *,
    verbose: bool = False,
 ) -> Tuple[Dict, int]:
    """Roll out actions in env and return trajectory dict in robomimic format.

    When `verbose=True`, the current `agentview_image` frame is shown with OpenCV.
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    env.reset()
    env.reset_to(initial_state)

    obs = env.get_observation()
    state_dict = env.get_state()

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        states=[],
        actions=actions,
        initial_state_dict=state_dict,
    )

    for t in range(1, states.shape[0] + 1):
        if reset_every_step and t < states.shape[0]:
            next_obs = env.reset_to({"states": states[t]})
        else:
            next_obs, _, _, _ = env.step(actions[t - 1])

        if verbose:
            im_viz = obs.get("agentview_image", None)
            if im_viz is not None:
                if np.issubdtype(im_viz.dtype, np.floating):
                    im_viz = np.clip(im_viz * 255.0, 0, 255).astype(np.uint8)
                im_viz = cv2.cvtColor(im_viz, cv2.COLOR_RGB2BGR)
                cv2.imshow("agentview", im_viz)
                cv2.waitKey(1)

        r = env.get_reward()
        done = env.is_success()["task"]
        done = int(done)

        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["states"].append(state_dict["states"])

        obs = deepcopy(next_obs)
        state_dict = env.get_state()

    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for sub_k in traj[k]:
                traj[k][sub_k] = np.array(traj[k][sub_k])
        else:
            traj[k] = np.array(traj[k])

    return traj, done
