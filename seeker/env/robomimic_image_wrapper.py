import re
from typing import List, Optional

import cv2
import gym
import numpy as np
from gym import spaces
from robomimic.envs.env_robosuite import EnvRobosuite
from robosuite.utils.camera_utils import (
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)

from mimicgen.configs import MG_TaskSpec, config_factory
from mimicgen.env_interfaces.base import make_interface

from seeker.dataset.oracle_affordance import OracleAffordanceResolver


def _normalize_mimicgen_task_name(env_name: str) -> str:
    """Map env/task names like ``Square_D0`` or ``square_d2`` to ``square``."""
    name = str(env_name).strip()
    name = re.sub(r"(_)?(d\d+|v\d+)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"_+", "_", name)
    return name.lower()


def _default_interface_name(task_name: str) -> str:
    """Return MimicGen's conventional robosuite interface class name."""
    return "MG_" + "".join(part.capitalize() for part in task_name.split("_"))


def _build_task_spec(task_name: str) -> MG_TaskSpec:
    """Build a MimicGen task spec from the registered robosuite config."""
    mg_config = config_factory(name=task_name, config_type="robosuite")
    return MG_TaskSpec.from_json(json_string=mg_config.task.task_spec.dump())


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key="agentview_image",
        enable_oracle_subtask_info: bool = False,
        oracle_task_name: Optional[str] = None,
        oracle_interface_name: Optional[str] = None,
        oracle_interface_type: str = "robosuite",
        oracle_projection_camera: Optional[str] = None,
        oracle_projection_height: Optional[int] = None,
        oracle_projection_width: Optional[int] = None,
        enable_oracle_focus_info: bool = False,
        oracle_focus_camera: Optional[str] = None,
        oracle_focus_resolution: Optional[int] = None,
        oracle_focus_patch_size: int = 16,
        oracle_focus_min_patch_area_fraction: float = 0.05,
        oracle_focus_min_mask_pixels: int = 16,
        enable_oracle_video_overlay: bool = False,
        oracle_overlay_zoom: float = 4.0,
    ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        self.enable_oracle_subtask_info = bool(enable_oracle_subtask_info)
        self.oracle_projection_camera = oracle_projection_camera
        self.oracle_projection_height = oracle_projection_height
        self.oracle_projection_width = oracle_projection_width
        self.enable_oracle_focus_info = bool(enable_oracle_focus_info)
        self.oracle_focus_camera = oracle_focus_camera or oracle_projection_camera
        self.oracle_focus_resolution = oracle_focus_resolution
        self.oracle_focus_patch_size = int(oracle_focus_patch_size)
        self.oracle_focus_min_patch_area_fraction = float(
            oracle_focus_min_patch_area_fraction
        )
        self.oracle_focus_min_mask_pixels = int(oracle_focus_min_mask_pixels)
        self.enable_oracle_video_overlay = bool(enable_oracle_video_overlay)
        self.oracle_overlay_zoom = float(oracle_overlay_zoom)
        self.prediction_video_boxes = []
        self.oracle_env_interface = None
        self.oracle_task_spec = None
        self.oracle_affordance = None
        self.oracle_active_subtask_idx = 0
        self.oracle_prev_subtask_signals = {}
        self.oracle_cached_info = {}
        if self.enable_oracle_subtask_info or self.enable_oracle_focus_info:
            task_name = oracle_task_name or _normalize_mimicgen_task_name(env.name)
            task_name = _normalize_mimicgen_task_name(task_name)
            interface_name = oracle_interface_name or _default_interface_name(task_name)
            self.oracle_task_spec = _build_task_spec(task_name)
            self.oracle_env_interface = make_interface(
                name=interface_name,
                interface_type=oracle_interface_type,
                env=env.env,
            )
            if self.enable_oracle_focus_info:
                if self.oracle_focus_camera is None:
                    raise ValueError(
                        "oracle_focus_camera or oracle_projection_camera is required "
                        "when enable_oracle_focus_info=True"
                    )
                self.oracle_affordance = OracleAffordanceResolver(
                    env=env,
                    env_meta={"env_name": env.name},
                    camera_name=self.oracle_focus_camera,
                    resolution=self._oracle_focus_resolution(),
                    patch_size=self.oracle_focus_patch_size,
                )

        # setup spaces
        action_shape = shape_meta["action"]["shape"]
        action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_value, max_value = -1, 1
            if key.endswith("image"):
                min_value, max_value = 0, 1
            elif key.endswith("image_tcp_centered"):
                min_value, max_value = 0, 1
            elif key.endswith("depth"):
                min_value, max_value = 0, 1
            elif key.endswith("voxels"):
                min_value, max_value = 0, 1
            elif key.endswith("point_cloud"):
                min_value, max_value = -10, 10
            elif key.endswith("quat") or key.endswith("rot") or key.endswith("rot_6d"):
                min_value, max_value = -1, 1
            elif key.endswith("qpos"):
                min_value, max_value = -1, 1
            elif key.endswith("pos") or key.endswith("eef_z"):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")

            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()["states"]
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        self.oracle_active_subtask_idx = 0
        self.oracle_prev_subtask_signals = {}
        self.prediction_video_boxes = []
        self._update_oracle_cached_info(include_focus=self.enable_oracle_focus_info)
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        if self.enable_oracle_subtask_info or self.enable_oracle_focus_info:
            info = dict(info)
            info.update(
                self._update_oracle_cached_info(
                    include_focus=self.enable_oracle_focus_info
                )
            )
        return obs, reward, done, info

    def get_oracle_subtask_info(self, include_focus: Optional[bool] = None):
        """Return cached MimicGen oracle subtask object-ref and projection info."""
        _ = include_focus
        return dict(self.oracle_cached_info)

    def _update_oracle_cached_info(self, include_focus: Optional[bool] = None):
        """Advance oracle state once for the current env state and cache the result."""
        if self.oracle_env_interface is None or self.oracle_task_spec is None:
            self.oracle_cached_info = {}
            return {}
        if include_focus is None:
            include_focus = self.enable_oracle_focus_info

        datagen_info = self.oracle_env_interface.get_datagen_info(action=None)
        self.oracle_active_subtask_idx = self._advance_oracle_subtask(
            self.oracle_active_subtask_idx,
            self.oracle_prev_subtask_signals,
            datagen_info.subtask_term_signals,
        )
        self.oracle_prev_subtask_signals = self._scalar_signal_dict(
            datagen_info.subtask_term_signals
        )
        subtask_idx = self.oracle_active_subtask_idx
        subtask = self.oracle_task_spec[subtask_idx]
        object_ref = subtask["object_ref"]

        result = {
            "oracle_subtask_idx": np.asarray(subtask_idx, dtype=np.int64),
            "oracle_subtask_term_signal": subtask["subtask_term_signal"] or "",
            "oracle_object_ref": object_ref or "",
        }
        if object_ref is None:
            if include_focus:
                result.update(self._empty_oracle_focus_info())
            self.oracle_cached_info = result
            return result

        object_pose = datagen_info.object_poses[object_ref]
        xyz = np.asarray(object_pose[:3, 3], dtype=np.float32)
        result["oracle_object_xyz"] = xyz

        if self.oracle_projection_camera is not None:
            h, w = self._oracle_projection_size()
            world_to_pixel = get_camera_transform_matrix(
                sim=self.env.env.sim,
                camera_name=self.oracle_projection_camera,
                camera_height=h,
                camera_width=w,
            )
            row_col = project_points_from_world_to_camera(
                xyz[None],
                world_to_pixel,
                camera_height=h,
                camera_width=w,
            )[0].astype(np.int64)
            result["oracle_object_pixel"] = row_col
        if include_focus:
            result.update(
                self._oracle_focus_info(
                    object_ref=object_ref,
                    subtask_idx=subtask_idx,
                    object_xyz=xyz,
                )
            )
        self.oracle_cached_info = result
        return result

    def get_oracle_focus_info(self):
        """Return only live oracle focus arrays for the current env state."""
        info = self.oracle_cached_info
        return {
            key: value
            for key, value in info.items()
            if key.startswith("oracle_target_")
        }

    def _oracle_focus_resolution(self) -> int:
        if self.oracle_focus_resolution is not None:
            return int(self.oracle_focus_resolution)
        if self.oracle_focus_camera is not None:
            h, w = self._oracle_projection_size_for_camera(self.oracle_focus_camera)
            if h != w:
                raise ValueError(
                    f"Oracle focus expects square camera images, got {h}x{w}"
                )
            return int(h)
        return 256

    def _oracle_patch_grid_size(self) -> int:
        return int(
            np.ceil(
                float(self._oracle_focus_resolution())
                / float(max(self.oracle_focus_patch_size, 1))
            )
        )

    def _empty_oracle_focus_info(self) -> dict:
        if self.oracle_focus_camera is None:
            camera = "agentview"
        else:
            camera = str(self.oracle_focus_camera)
        grid_size = self._oracle_patch_grid_size()
        return {
            f"oracle_target_box_{camera}": np.full((4,), np.nan, dtype=np.float32),
            f"oracle_target_patch_mask_{camera}": np.zeros(
                (grid_size, grid_size), dtype=np.uint8
            ),
            f"oracle_target_mask_area_{camera}": np.asarray(np.nan, dtype=np.float32),
            "oracle_target_xyz": np.full((3,), np.nan, dtype=np.float32),
        }

    def _oracle_focus_info(self, *, object_ref: str, subtask_idx: int, object_xyz):
        if self.oracle_affordance is None or self.oracle_focus_camera is None:
            return self._empty_oracle_focus_info()
        spec = self.oracle_affordance.affordance_spec(
            ref=object_ref,
            subtask_idx=int(subtask_idx),
        )
        points = self.oracle_affordance.affordance_points(
            ref=object_ref,
            subtask_idx=int(subtask_idx),
            object_xyz=np.asarray(object_xyz, dtype=np.float32),
            spec=spec,
        )
        out = self._empty_oracle_focus_info()
        out["oracle_target_xyz"] = np.mean(
            np.asarray(points, dtype=np.float32).reshape(-1, 3),
            axis=0,
        ).astype(np.float32)
        target_box, target_area, target_mask = self.oracle_affordance.segmentation_box(
            ref=object_ref,
            spec=spec,
            min_patch_area_fraction=self.oracle_focus_min_patch_area_fraction,
            min_mask_pixels=self.oracle_focus_min_mask_pixels,
        )
        camera = str(self.oracle_focus_camera)
        if target_box is not None:
            out[f"oracle_target_box_{camera}"] = np.asarray(
                target_box, dtype=np.float32
            ).reshape(4)
            out[f"oracle_target_mask_area_{camera}"] = np.asarray(
                float(target_area), dtype=np.float32
            )
        if target_mask is not None:
            out[f"oracle_target_patch_mask_{camera}"] = np.asarray(
                target_mask, dtype=np.uint8
            )
        return out

    def _advance_oracle_subtask(self, current_idx, prev_signals, subtask_term_signals):
        """Advance on 0->1 subtask completion transitions."""
        idx = int(current_idx)
        while idx < len(self.oracle_task_spec) - 1:
            signal = self.oracle_task_spec[idx]["subtask_term_signal"]
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
    def _scalar_signal_dict(subtask_term_signals):
        return {
            key: int(np.asarray(value).reshape(-1)[0])
            for key, value in subtask_term_signals.items()
        }

    def _oracle_projection_size(self):
        return self._oracle_projection_size_for_camera(self.oracle_projection_camera)

    def _oracle_projection_size_for_camera(self, camera_name: str):
        h = self.oracle_projection_height
        w = self.oracle_projection_width
        if h is None or w is None:
            shape = self.shape_meta["obs"][f"{camera_name}_image"]["shape"]
            h = int(shape[-2])
            w = int(shape[-1])
        return int(h), int(w)

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        if self.enable_oracle_video_overlay:
            img = self._draw_oracle_overlay(img)
        if self.prediction_video_boxes:
            img = self._draw_prediction_overlay(img)
        return img

    def set_prediction_video_boxes(self, boxes):
        """Set per-frame predicted boxes to draw on recorded rollout videos."""
        self.prediction_video_boxes = [] if boxes is None else list(boxes)

    @staticmethod
    def _view_name_from_obs_key(obs_key: str) -> str:
        key = str(obs_key)
        if "eye_in_hand" in key or "in_hand" in key:
            return "eye_in_hand"
        if "agentview" in key:
            return "agentview"
        if key.endswith("_image"):
            return key[: -len("_image")]
        return key

    def _draw_oracle_overlay(self, img: np.ndarray) -> np.ndarray:
        """Draw an oracle object-ref box centered at its projected image point."""
        if self.oracle_projection_camera is None:
            return img
        if self.oracle_overlay_zoom <= 0:
            return img

        info = self.get_oracle_subtask_info()
        pixel = info.get("oracle_object_pixel", None)
        if pixel is None:
            return img

        out = img.copy()
        frame_h, frame_w = out.shape[:2]
        proj_h, proj_w = self._oracle_projection_size()
        row = int(round(float(pixel[0]) * frame_h / max(proj_h, 1)))
        col = int(round(float(pixel[1]) * frame_w / max(proj_w, 1)))

        box_h = max(2, int(round(frame_h / self.oracle_overlay_zoom)))
        box_w = max(2, int(round(frame_w / self.oracle_overlay_zoom)))
        r0 = max(0, row - box_h // 2)
        r1 = min(frame_h - 1, row + box_h // 2)
        c0 = max(0, col - box_w // 2)
        c1 = min(frame_w - 1, col + box_w // 2)

        color = np.asarray([0, 255, 80], dtype=np.uint8)
        thickness = max(2, int(round(min(frame_h, frame_w) / 128)))
        out[r0 : min(r0 + thickness, frame_h), c0 : c1 + 1] = color
        out[max(r1 - thickness + 1, 0) : r1 + 1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0 : min(c0 + thickness, frame_w)] = color
        out[r0 : r1 + 1, max(c1 - thickness + 1, 0) : c1 + 1] = color

        dot = max(2, thickness + 1)
        out[
            max(0, row - dot) : min(frame_h, row + dot + 1),
            max(0, col - dot) : min(frame_w, col + dot + 1),
        ] = color
        return out

    def _draw_prediction_overlay(self, img: np.ndarray) -> np.ndarray:
        """Draw predicted focus boxes / points in render-frame pixels."""
        out = img.copy()
        frame_h, frame_w = out.shape[:2]
        render_view = self._view_name_from_obs_key(self.render_obs_key)
        colors = {
            "seeker": (0, 220, 255),
            "rvt2_heatmap": (255, 210, 0),
            "oracle": (0, 255, 80),
            "focus_refiner": (0, 220, 255),
        }
        point_colors = [
            (255, 64, 64),
            (64, 192, 255),
            (255, 220, 64),
            (96, 255, 128),
            (224, 96, 255),
            (255, 144, 64),
            (64, 255, 224),
            (160, 160, 255),
        ]
        thickness = max(2, int(round(min(frame_h, frame_w) / 128)))

        for item in self.prediction_video_boxes:
            if not isinstance(item, dict):
                continue
            item_view = item.get("view")
            if item_view is not None and str(item_view) != render_view:
                continue

            source = str(item.get("source", "pred"))
            color = colors.get(source, (255, 80, 80))
            source_size = max(
                float(item.get("source_size", max(frame_h, frame_w))),
                1.0,
            )
            x_scale = float(frame_w - 1) / max(source_size - 1.0, 1.0)
            y_scale = float(frame_h - 1) / max(source_size - 1.0, 1.0)

            if "box_px" in item:
                self._draw_prediction_box(
                    out,
                    box_px=item["box_px"],
                    label=source.upper(),
                    color=color,
                    thickness=thickness,
                    x_scale=x_scale,
                    y_scale=y_scale,
                )

            if "points_px" in item:
                self._draw_prediction_points(
                    out,
                    points_px=item["points_px"],
                    point_colors=point_colors,
                    mean_point_index=item.get("mean_point_index"),
                    radius=max(2, thickness + 1),
                    x_scale=x_scale,
                    y_scale=y_scale,
                )

        return out

    @staticmethod
    def _draw_prediction_box(
        out: np.ndarray,
        *,
        box_px,
        label: str,
        color: tuple[int, int, int],
        thickness: int,
        x_scale: float,
        y_scale: float,
    ) -> None:
        frame_h, frame_w = out.shape[:2]
        box = np.asarray(box_px, dtype=np.float32).reshape(-1)
        if box.shape[0] != 4 or not np.all(np.isfinite(box)):
            return

        x0, y0 = RobomimicImageWrapper._scale_overlay_point(
            box[:2],
            x_scale=x_scale,
            y_scale=y_scale,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        x1, y1 = RobomimicImageWrapper._scale_overlay_point(
            box[2:],
            x_scale=x_scale,
            y_scale=y_scale,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness=thickness)
        cv2.putText(
            out,
            label,
            (x0, max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _draw_prediction_points(
        out: np.ndarray,
        *,
        points_px,
        point_colors: list[tuple[int, int, int]],
        mean_point_index,
        radius: int,
        x_scale: float,
        y_scale: float,
    ) -> None:
        frame_h, frame_w = out.shape[:2]
        points = np.asarray(points_px, dtype=np.float32).reshape(-1, 2)
        mean_idx = None if mean_point_index is None else int(mean_point_index)
        for idx, point in enumerate(points):
            if not np.all(np.isfinite(point)):
                continue
            x, y = RobomimicImageWrapper._scale_overlay_point(
                point,
                x_scale=x_scale,
                y_scale=y_scale,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            if mean_idx is not None and idx == mean_idx:
                RobomimicImageWrapper._draw_square(
                    out,
                    x=x,
                    y=y,
                    radius=radius + 2,
                    color=(0, 0, 0),
                )
                RobomimicImageWrapper._draw_square(
                    out,
                    x=x,
                    y=y,
                    radius=radius,
                    color=(255, 255, 255),
                )
                continue

            RobomimicImageWrapper._draw_square(
                out,
                x=x,
                y=y,
                radius=radius,
                color=point_colors[idx % len(point_colors)],
            )

    @staticmethod
    def _scale_overlay_point(
        point,
        *,
        x_scale: float,
        y_scale: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int]:
        x = int(round(float(point[0]) * x_scale))
        y = int(round(float(point[1]) * y_scale))
        x = int(np.clip(x, 0, frame_w - 1))
        y = int(np.clip(y, 0, frame_h - 1))
        return x, y

    @staticmethod
    def _draw_square(
        out: np.ndarray,
        *,
        x: int,
        y: int,
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        frame_h, frame_w = out.shape[:2]
        cv2.rectangle(
            out,
            (max(0, x - radius), max(0, y - radius)),
            (min(frame_w - 1, x + radius), min(frame_h - 1, y + radius)),
            color,
            thickness=-1,
        )
