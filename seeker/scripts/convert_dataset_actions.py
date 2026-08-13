"""Helpers for converting robomimic actions between controller representations."""

from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

def convert_actions(
    env,
    states: np.ndarray,
    actions: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Convert source actions into absolute controller action space.

    Args:
        env: Robomimic environment whose controller interprets ``actions``.
        states: Simulator states aligned one-to-one with ``actions``.
        actions: Controller actions with final dimension ``7 * num_robots``.

    Returns:
        ``(converted_actions, robot_eef_rots)`` where ``converted_actions`` has
        an ``"absolute"`` entry in the same shape as ``actions``, and
        ``robot_eef_rots`` is ``[T, num_robots, 9]``.
    """
    actions = actions.copy()

    stacked_actions = actions.reshape(*actions.shape[:-1], -1, 7).astype(np.float64)
    num_frames, num_robots = stacked_actions.shape[:2]
    if num_robots != 1:
        raise NotImplementedError(
            "Multi-robot action conversion is not supported: "
            f"{stacked_actions.shape}"
        )

    abs_goal_pos = np.zeros((num_frames, num_robots, 3), dtype=np.float64)
    abs_goal_ori = np.zeros((num_frames, num_robots, 3), dtype=np.float64)
    action_gripper = stacked_actions[..., [-1]]

    robot_eef_rots = np.zeros((num_frames, num_robots, 9), dtype=np.float64)

    for i in range(num_frames):
        env.reset_to({"states": states[i]})
        for idx, robot in enumerate(env.env.robots):
            robot.control(stacked_actions[i, idx], policy_step=True)
            controller = robot.controller

            goal_pos, goal_ori = controller.goal_pos, controller.goal_ori
            ee_ori_mat = controller.ee_ori_mat

            abs_goal_pos[i, idx] = goal_pos
            abs_goal_ori[i, idx] = Rotation.from_matrix(goal_ori).as_rotvec()

            robot_eef_rots[i, idx] = ee_ori_mat.flatten()

    abs_actions = np.concatenate(
        [abs_goal_pos, abs_goal_ori, action_gripper], axis=-1
    ).reshape(actions.shape)
    converted_actions = {
        "absolute": abs_actions.astype(np.float32),
    }
    return converted_actions, robot_eef_rots
