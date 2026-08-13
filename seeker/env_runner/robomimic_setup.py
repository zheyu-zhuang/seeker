"""Robomimic environment construction for rollout runners."""

from __future__ import annotations

import collections
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import dill
import mimicgen  # noqa
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from seeker.env.robomimic_image_wrapper import RobomimicImageWrapper
from seeker.env_runner.async_vector_env import AsyncVectorEnv
from seeker.env_runner.multistep_wrapper import MultiStepWrapper
from seeker.env_runner.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from seeker.env.mjcf_texture import list_texture_files
from seeker.env.robomimic import (
    install_table_texture_hook,
    table_texture,
    update_env_controller,
)
from seeker.dataset.action import validate_action_rep
from seeker.dataset.cache import load_metadata, resolve_cache_dir
from seeker import TEXTURES_DIR


install_table_texture_hook()


@dataclass
class RobomimicRunnerSetup:
    env_meta: dict
    action_rep: str
    env: AsyncVectorEnv
    env_fns: list
    env_seeds: list[int]
    env_prefixes: list[str]
    env_init_fn_dills: list[bytes]
    env_video_enabled: list[bool]
    default_shape_meta: dict
    oracle_projection_camera: str


def assign_textures(texture_dir, eval_split=True, n_envs=8, seed=0):
    split_dir = Path(texture_dir) / ("eval" if eval_split else "train")
    files = list_texture_files(split_dir)
    if len(files) == 0:
        raise RuntimeError(f"No textures found in {split_dir}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    shuffled = [files[i] for i in order]

    return [shuffled[i % len(shuffled)] for i in range(n_envs)]


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )


def default_rollout_shape_meta(default_res: int = 256) -> dict:
    return {
        "obs": {
            "agentview_image": {
                "shape": [3, default_res, default_res],
                "type": "rgb",
            },
            "robot0_eye_in_hand_image": {
                "shape": [3, default_res, default_res],
                "type": "rgb",
            },
            "robot0_eef_pos": {
                "shape": [3],
            },
            "robot0_eef_rot": {
                "shape": [9],
            },
            "robot0_gripper_qpos": {
                "shape": [2],
            },
        },
        "action": {
            "shape": [10],
        },
    }


def _resolve_oracle_projection_camera(
    *, oracle_projection_camera: Optional[str], render_obs_key: str
) -> str:
    if oracle_projection_camera is not None:
        return oracle_projection_camera
    camera = render_obs_key
    if camera.endswith("_image"):
        camera = camera[: -len("_image")]
    return camera


def _wrap_robomimic_env(
    *,
    robomimic_env,
    shape_meta: dict,
    render_obs_key: str,
    fps: int,
    crf: int,
    steps_per_render: int,
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    enable_oracle_subtask_info: bool,
    oracle_task_name: Optional[str] = None,
    oracle_projection_camera: Optional[str] = None,
    oracle_projection_height: Optional[int] = None,
    oracle_projection_width: Optional[int] = None,
    enable_oracle_focus_info: bool = False,
    oracle_focus_camera: Optional[str] = None,
    oracle_focus_patch_size: int = 16,
    oracle_focus_min_patch_area_fraction: float = 0.05,
    oracle_focus_min_mask_pixels: int = 16,
    enable_oracle_video_overlay: bool = False,
    oracle_overlay_zoom: float = 4.0,
):
    return MultiStepWrapper(
        VideoRecordingWrapper(
            RobomimicImageWrapper(
                env=robomimic_env,
                shape_meta=shape_meta,
                init_state=None,
                render_obs_key=render_obs_key,
                enable_oracle_subtask_info=enable_oracle_subtask_info,
                oracle_task_name=oracle_task_name,
                oracle_projection_camera=oracle_projection_camera,
                oracle_projection_height=oracle_projection_height,
                oracle_projection_width=oracle_projection_width,
                enable_oracle_focus_info=enable_oracle_focus_info,
                oracle_focus_camera=oracle_focus_camera,
                oracle_focus_resolution=oracle_projection_height,
                oracle_focus_patch_size=oracle_focus_patch_size,
                oracle_focus_min_patch_area_fraction=(
                    oracle_focus_min_patch_area_fraction
                ),
                oracle_focus_min_mask_pixels=oracle_focus_min_mask_pixels,
                enable_oracle_video_overlay=enable_oracle_video_overlay,
                oracle_overlay_zoom=oracle_overlay_zoom,
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=crf,
                thread_type="FRAME",
                thread_count=1,
            ),
            file_path=None,
            steps_per_render=steps_per_render,
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
    )


def _build_vector_env(
    *,
    env_meta: dict,
    default_shape_meta: dict,
    n_envs: int,
    render_obs_key: str,
    fps: int,
    crf: int,
    steps_per_render: int,
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    shuffle_table_texture: bool,
    enable_oracle_subtask_info: bool,
    oracle_projection_camera: str,
    enable_oracle_focus_info: bool,
    oracle_focus_camera: str,
    oracle_focus_patch_size: int,
    oracle_focus_min_patch_area_fraction: float,
    oracle_focus_min_mask_pixels: int,
    default_res: int,
    enable_oracle_video_overlay: bool,
    oracle_overlay_zoom: float,
):
    assigned = (
        assign_textures(str(TEXTURES_DIR), eval_split=True, n_envs=n_envs, seed=0)
        if shuffle_table_texture
        else [None] * n_envs
    )

    def make_env_fn(idx: int):
        tex = assigned[idx]

        def env_fn():
            with table_texture(tex):
                robomimic_env = create_env(
                    env_meta=env_meta, shape_meta=default_shape_meta
                )
                robomimic_env.env.hard_reset = False

            return _wrap_robomimic_env(
                robomimic_env=robomimic_env,
                shape_meta=default_shape_meta,
                render_obs_key=render_obs_key,
                fps=fps,
                crf=crf,
                steps_per_render=steps_per_render,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_steps=max_steps,
                enable_oracle_subtask_info=enable_oracle_subtask_info,
                oracle_task_name=env_meta["env_name"],
                oracle_projection_camera=oracle_projection_camera,
                oracle_projection_height=default_res,
                oracle_projection_width=default_res,
                enable_oracle_focus_info=enable_oracle_focus_info,
                oracle_focus_camera=oracle_focus_camera,
                oracle_focus_patch_size=oracle_focus_patch_size,
                oracle_focus_min_patch_area_fraction=(
                    oracle_focus_min_patch_area_fraction
                ),
                oracle_focus_min_mask_pixels=oracle_focus_min_mask_pixels,
                enable_oracle_video_overlay=enable_oracle_video_overlay,
                oracle_overlay_zoom=oracle_overlay_zoom,
            )

        return env_fn

    def dummy_env_fn():
        robomimic_env = create_env(
            env_meta=env_meta, shape_meta=default_shape_meta, enable_render=False
        )
        return _wrap_robomimic_env(
            robomimic_env=robomimic_env,
            shape_meta=default_shape_meta,
            render_obs_key=render_obs_key,
            fps=fps,
            crf=crf,
            steps_per_render=steps_per_render,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_steps=max_steps,
            enable_oracle_subtask_info=False,
            enable_oracle_focus_info=False,
        )

    env_fns = [make_env_fn(i) for i in range(n_envs)]
    return AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn), env_fns


def _build_rollout_init_specs(
    *,
    output_dir: str,
    n_test: int,
    n_test_vis: int,
    test_start_seed: int,
) -> tuple[list[int], list[str], list[bytes], list[bool]]:
    env_seeds: list[int] = []
    env_prefixes: list[str] = []
    env_init_fn_dills: list[bytes] = []
    env_video_enabled: list[bool] = []

    for i in range(n_test):
        seed = test_start_seed + i
        enable_render = i < n_test_vis

        def init_fn(env, seed=seed, enable_render=enable_render):
            assert isinstance(env.env, VideoRecordingWrapper)
            env.env.video_recoder.stop()
            env.env.file_path = None
            if enable_render:
                filename = Path(output_dir).joinpath(
                    "media",
                    f"test_seed_{seed}.mp4",
                )
                filename.parent.mkdir(parents=False, exist_ok=True)
                env.env.file_path = str(filename)

            assert isinstance(env.env.env, RobomimicImageWrapper)
            env.env.env.init_state = None
            env.seed(seed)

        env_seeds.append(seed)
        env_prefixes.append("test/")
        env_init_fn_dills.append(dill.dumps(init_fn))
        env_video_enabled.append(enable_render)

    return env_seeds, env_prefixes, env_init_fn_dills, env_video_enabled


def _load_env_meta_from_cache(
    *,
    dataset_path: str,
    cache_dir: Optional[str],
    env_name: Optional[str],
) -> dict:
    cache_path = resolve_cache_dir(dataset_path, cache_dir)
    meta, _ = load_metadata(str(cache_path))
    if isinstance(meta.get("env_meta"), dict):
        env_meta = copy.deepcopy(meta["env_meta"])
        if env_name is None or str(env_meta.get("env_name", "")) == str(env_name):
            return env_meta

    source_env_metas = meta.get("source_env_metas", [])
    matches = []
    for item in source_env_metas:
        if not isinstance(item, dict):
            continue
        if env_name is None or str(item.get("env_name", "")) == str(env_name):
            matches.append(item)

    if len(matches) == 1:
        return copy.deepcopy(matches[0])

    available = [
        str(item.get("env_name", ""))
        for item in source_env_metas
        if isinstance(item, dict)
    ]
    if env_name is None:
        raise ValueError(
            "Merged cache contains multiple env_metas; set env_name to select "
            f"one. Available envs: {available}"
        )
    raise ValueError(
        f"Cache metadata does not contain env_meta for env_name={env_name!r}. "
        f"Available envs: {available}"
    )


def build_robomimic_runner_setup(
    *,
    output_dir: str,
    dataset_path: str,
    cache_dir: Optional[str],
    action_rep: str,
    n_test: int,
    n_test_vis: int,
    test_start_seed: int,
    max_steps: int,
    n_obs_steps: int,
    n_action_steps: int,
    render_obs_key: str,
    fps: int,
    crf: int,
    n_envs: int,
    env_name: Optional[str],
    shuffle_table_texture: bool,
    enable_oracle_subtask_info: bool,
    oracle_projection_camera: Optional[str],
    enable_oracle_focus_info: bool = False,
    oracle_focus_camera: Optional[str] = None,
    oracle_focus_patch_size: int = 16,
    oracle_focus_min_patch_area_fraction: float = 0.05,
    oracle_focus_min_mask_pixels: int = 16,
    enable_oracle_video_overlay: bool = False,
    oracle_overlay_zoom: float = 4.0,
) -> RobomimicRunnerSetup:
    dataset_path = os.path.expanduser(dataset_path)
    action_rep = validate_action_rep(action_rep)

    env_meta = _load_env_meta_from_cache(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        env_name=env_name,
    )
    env_meta = update_env_controller(env_meta, action_rep)
    env_meta["env_kwargs"]["use_object_obs"] = False
    if env_name is not None:
        env_meta["env_name"] = env_name

    default_res = 256
    env_meta["env_kwargs"]["camera_heights"] = default_res
    env_meta["env_kwargs"]["camera_widths"] = default_res
    default_shape_meta = default_rollout_shape_meta(default_res)
    oracle_projection_camera = _resolve_oracle_projection_camera(
        oracle_projection_camera=oracle_projection_camera,
        render_obs_key=render_obs_key,
    )
    oracle_focus_camera = _resolve_oracle_projection_camera(
        oracle_projection_camera=oracle_focus_camera or oracle_projection_camera,
        render_obs_key=render_obs_key,
    )

    robosuite_fps = 20
    steps_per_render = max(robosuite_fps // fps, 1)
    env, env_fns = _build_vector_env(
        env_meta=env_meta,
        default_shape_meta=default_shape_meta,
        n_envs=n_envs,
        render_obs_key=render_obs_key,
        fps=fps,
        crf=crf,
        steps_per_render=steps_per_render,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
        shuffle_table_texture=shuffle_table_texture,
        enable_oracle_subtask_info=enable_oracle_subtask_info,
        oracle_projection_camera=oracle_projection_camera,
        enable_oracle_focus_info=enable_oracle_focus_info,
        oracle_focus_camera=oracle_focus_camera,
        oracle_focus_patch_size=oracle_focus_patch_size,
        oracle_focus_min_patch_area_fraction=oracle_focus_min_patch_area_fraction,
        oracle_focus_min_mask_pixels=oracle_focus_min_mask_pixels,
        default_res=default_res,
        enable_oracle_video_overlay=enable_oracle_video_overlay,
        oracle_overlay_zoom=oracle_overlay_zoom,
    )
    env_seeds, env_prefixes, env_init_fn_dills, env_video_enabled = (
        _build_rollout_init_specs(
            output_dir=output_dir,
            n_test=n_test,
            n_test_vis=n_test_vis,
            test_start_seed=test_start_seed,
        )
    )
    return RobomimicRunnerSetup(
        env_meta=env_meta,
        action_rep=action_rep,
        env=env,
        env_fns=env_fns,
        env_seeds=env_seeds,
        env_prefixes=env_prefixes,
        env_init_fn_dills=env_init_fn_dills,
        env_video_enabled=env_video_enabled,
        default_shape_meta=default_shape_meta,
        oracle_projection_camera=oracle_projection_camera,
    )
