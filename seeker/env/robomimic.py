"""Robomimic/robosuite integration helpers for controllers, textures, and rollouts."""

from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
import os
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.envs.env_base import EnvBase
from scipy.spatial.transform import Rotation as R
from seeker.dataset.action import validate_action_rep
from seeker.env.mjcf_texture import apply_table_texture

import cv2

_TABLE_TEXTURE_FILE: ContextVar[Optional[str]] = ContextVar(
    "_TABLE_TEXTURE_FILE", default=None
)
_HOOK_INSTALLED = False


def camera_pose_from_xml(pos, quat):
    """Convert XML camera pose (pos, wxyz quat) to world transform matrix."""
    X_C = np.eye(4)
    obj_quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    X_C[:3, :3] = R.from_quat(obj_quat_xyzw).as_matrix()
    X_C[:3, 3] = pos
    camera_axis_correction = np.diag([1, -1, -1, 1])
    return X_C @ camera_axis_correction


def require_upstream_controller_config(env_meta):
    """Validate controller metadata for upstream robosuite OSC configs."""
    controller_configs = env_meta["env_kwargs"]["controller_configs"]
    if "action_mode" in controller_configs:
        raise ValueError(
            "Legacy patched OSC key 'action_mode' is no longer supported. "
            "Use upstream robosuite controller metadata with 'control_delta'."
        )
    return env_meta


def update_env_controller(env_meta, action_rep):
    """Patch controller config so execution uses absolute OSC targets."""
    validate_action_rep(action_rep)

    env_meta = deepcopy(env_meta)
    require_upstream_controller_config(env_meta)
    controller_configs = env_meta["env_kwargs"]["controller_configs"]
    controller_configs["control_delta"] = False
    return env_meta


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
    pre_step_callback=None,
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

    for t in range(states.shape[0]):
        if reset_every_step:
            if t == 0:
                env.reset_to(initial_state)
            else:
                env.reset_to({"states": states[t]})
            obs = env.get_observation()
            state_dict = env.get_state()

        if pre_step_callback is not None:
            pre_step_callback(env=env, t=t, obs=obs, state_dict=state_dict)

        next_obs, _, _, _ = env.step(actions[t])

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

        if not reset_every_step:
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

    success = int(np.max(traj["dones"])) if traj["dones"].size > 0 else int(done)
    return traj, success
