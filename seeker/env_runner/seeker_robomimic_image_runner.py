import collections
import math
import os
import pathlib
from typing import Optional

import dill
import numpy as np
import torch
import tqdm
from torch.utils._pytree import tree_map

from seeker.env_runner.robomimic_setup import build_robomimic_runner_setup
from seeker.env_runner.rollout_media import (
    extract_agentview_start_frames,
    extract_policy_prediction_video_boxes,
    rollout_snapshot_filename,
    rollout_video_filename,
    save_image_grid,
)
from seeker.env_runner.video_recording_wrapper import VideoRecordingWrapper
from seeker.model.common.rotation_transformer import RotationTransformer
from seeker.policy.base_image_policy import BaseImagePolicy
from seeker.util.formatting import find_all_keys
from seeker.util.mirror_augmentation import (
    MirrorAugmentationConfig,
    action_mirror_frame_to_world,
    center_lowdim_observations,
)
from seeker.util.task_meta import (
    env_name_to_robot_id,
    env_name_to_task_embedding,
    env_name_to_instruction,
    instruction_to_task_language_tokens,
    setup_task_embedding_cache,
)


class SeekerRobomimicImageRunner:
    def __init__(
        self,
        output_dir,
        dataset_path,
        shape_meta: dict,
        cache_dir=None,
        action_rep: str = "absolute",
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="agentview_image",
        fps=10,
        crf=22,
        past_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        env_name=None,
        shuffle_table_texture=False,
        strict_task_success=True,
        enable_oracle_subtask_info=True,
        oracle_projection_camera=None,
        enable_oracle_focus_info=False,
        oracle_focus_camera=None,
        oracle_focus_patch_size=16,
        oracle_focus_min_patch_area_fraction=0.05,
        oracle_focus_min_mask_pixels=16,
        enable_oracle_video_overlay=False,
        oracle_overlay_zoom=4.0,
        enable_prediction_video_overlay=False,
        mirror_augmentation=None,
    ):
        setup_task_embedding_cache()

        if n_envs is None:
            n_envs = n_test

        runner_setup = build_robomimic_runner_setup(
            output_dir=output_dir,
            dataset_path=dataset_path,
            cache_dir=cache_dir,
            action_rep=action_rep,
            n_test=n_test,
            n_test_vis=n_test_vis,
            test_start_seed=test_start_seed,
            max_steps=max_steps,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            render_obs_key=render_obs_key,
            fps=fps,
            crf=crf,
            n_envs=n_envs,
            env_name=env_name,
            shuffle_table_texture=shuffle_table_texture,
            enable_oracle_subtask_info=enable_oracle_subtask_info,
            oracle_projection_camera=oracle_projection_camera,
            enable_oracle_focus_info=enable_oracle_focus_info,
            oracle_focus_camera=oracle_focus_camera,
            oracle_focus_patch_size=oracle_focus_patch_size,
            oracle_focus_min_patch_area_fraction=oracle_focus_min_patch_area_fraction,
            oracle_focus_min_mask_pixels=oracle_focus_min_mask_pixels,
            enable_oracle_video_overlay=enable_oracle_video_overlay,
            oracle_overlay_zoom=oracle_overlay_zoom,
        )
        rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")
        delta_rotation_transformer = RotationTransformer("rotation_6d", "matrix")

        self.env_meta = runner_setup.env_meta
        self.action_rep = runner_setup.action_rep
        self.env = runner_setup.env
        self.env_fns = runner_setup.env_fns
        self.env_seeds = runner_setup.env_seeds
        self.env_prefixes = runner_setup.env_prefixes
        self.env_init_fn_dills = runner_setup.env_init_fn_dills
        self.env_video_enabled = runner_setup.env_video_enabled
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.delta_rotation_transformer = delta_rotation_transformer
        self.tqdm_interval_sec = tqdm_interval_sec
        self.max_rewards = {}
        self.shape_meta = shape_meta
        self.default_shape_meta = runner_setup.default_shape_meta
        self.output_dir = output_dir
        self.strict_task_success = strict_task_success
        self.enable_oracle_subtask_info = bool(enable_oracle_subtask_info)
        self.oracle_projection_camera = runner_setup.oracle_projection_camera
        self.enable_oracle_focus_info = bool(enable_oracle_focus_info)
        self.oracle_focus_camera = (
            runner_setup.oracle_projection_camera
            if oracle_focus_camera is None
            else str(oracle_focus_camera)
        )
        self.enable_oracle_video_overlay = bool(enable_oracle_video_overlay)
        self.oracle_overlay_zoom = float(oracle_overlay_zoom)
        self.enable_prediction_video_overlay = bool(enable_prediction_video_overlay)
        self.mirror_augmentation = MirrorAugmentationConfig.from_config(
            mirror_augmentation
        )
        self.pos_key = find_all_keys(
            "pos",
            self.default_shape_meta["obs"].keys(),
            only_one_match=True,
            reject=["delta", "qpos"],
        )
        self.rot_key = find_all_keys(
            "rot",
            self.default_shape_meta["obs"].keys(),
            only_one_match=True,
            reject=["delta"],
        )
        self.obs_rot_is_6d = self.shape_meta["obs"][self.rot_key]["shape"] == [6]
        env_name = self.env_meta["env_name"]
        self.task_embedding = env_name_to_task_embedding(env_name).cpu().numpy()
        self.task_embedding = self.task_embedding.astype(np.float32, copy=False)
        instruction = env_name_to_instruction(env_name)
        self.task_language_tokens = (
            instruction_to_task_language_tokens(instruction).cpu().numpy()
        )
        self.task_language_tokens = self.task_language_tokens.astype(
            np.float32,
            copy=False,
        )
        self.robot_id = int(env_name_to_robot_id(env_name))
        for prefix in self.env_prefixes:
            self.max_rewards[prefix] = 0
        self.max_rewards["total/"] = 0

    def run(self, policy: BaseImagePolicy, *, epoch: Optional[int] = None):
        device = policy.device
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_start_frames = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = []
            for global_idx in range(start, end):
                base_init_fn = self.env_init_fn_dills[global_idx]
                video_path = None
                if self.env_video_enabled[global_idx]:
                    video_path = pathlib.Path(self.output_dir).joinpath(
                        "media",
                        rollout_video_filename(
                            epoch=epoch,
                            prefix=self.env_prefixes[global_idx],
                            seed=self.env_seeds[global_idx],
                        ),
                    )
                    video_path.parent.mkdir(parents=False, exist_ok=True)
                    video_path = str(video_path)

                def init_fn(env, base_init_fn=base_init_fn, video_path=video_path):
                    fn = dill.loads(base_init_fn)
                    fn(env)
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = video_path

                this_init_fns.append(dill.dumps(init_fn))
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([this_init_fns[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            start_frames = extract_agentview_start_frames(obs)
            all_start_frames[this_global_slice] = list(start_frames[this_local_slice])
            past_action = None
            policy.reset()
            oracle_info = self._current_oracle_focus_info(env, n_envs)

            env_name = self.env_meta["env_name"]

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name} Image {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False

            while not done:
                # current eef position and rotation
                base_pos = obs[self.pos_key][:, -1].astype(np.float32)  # B, 3
                base_rot = obs[self.rot_key][:, -1].astype(np.float32)  # B, 9
                # convert obs
                self.preprocess_obs(obs)
                np_obs_dict = dict(obs)
                if oracle_info is not None:
                    np_obs_dict["oracle_info"] = oracle_info
                if self.past_action and (past_action is not None):
                    np_obs_dict["past_action"] = past_action[
                        :, -(self.n_obs_steps - 1) :
                    ].astype(np.float32)

                # device transfer
                obs_dict = tree_map(
                    lambda x: torch.from_numpy(x).to(device=device)
                    if isinstance(x, np.ndarray)
                    else x,
                    np_obs_dict,
                )

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                prediction_video_boxes = None
                if self.enable_prediction_video_overlay:
                    prediction_video_boxes = extract_policy_prediction_video_boxes(
                        policy,
                        n_envs=n_envs,
                    )

                # device_transfer
                np_action_dict = tree_map(
                    lambda x: x.detach().to("cpu").numpy()
                    if torch.is_tensor(x)
                    else x,
                    action_dict,
                )

                action = np_action_dict["action"]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action_input = action
                if self.mirror_augmentation.enable and self.action_rep == "absolute":
                    env_action_input = action_mirror_frame_to_world(
                        env_action_input,
                        self.mirror_augmentation,
                        action_rep=self.action_rep,
                        action_dim=self.shape_meta["action"]["shape"][0],
                    )
                env_action = self.undo_transform_action(
                    env_action_input, base_pos=base_pos, base_rot=base_rot
                )
                if prediction_video_boxes is not None:
                    env.call_each(
                        "set_prediction_video_boxes",
                        args_list=[(boxes,) for boxes in prediction_video_boxes],
                    )

                obs, reward, done, info = env.step(env_action)
                oracle_info = self._oracle_focus_info_from_step_infos(info)
                if oracle_info is None:
                    oracle_info = self._current_oracle_focus_info(env, n_envs)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        total_scores = []
        log_data = dict()
        rollout_success_flags = []
        rollout_labels = []

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixes[i]
            max_reward = np.max(all_rewards[i])
            episode_score = float(max_reward)
            if self.strict_task_success:
                episode_score = float(max_reward >= 1.0)

            max_rewards[prefix].append(episode_score)
            total_scores.append(episode_score)
            success = bool(episode_score >= 1.0)
            rollout_success_flags.append(success)
            rollout_labels.append(
                f"{prefix.strip('/')} {seed} {'SUCCESS' if success else 'FAIL'}"
            )
            if self.strict_task_success:
                log_data[prefix + f"sim_task_success_{seed}"] = episode_score

        snapshot_items = [
            (frame, success, label)
            for frame, success, label in zip(
                all_start_frames,
                rollout_success_flags,
                rollout_labels,
            )
            if frame is not None
        ]
        if snapshot_items:
            snapshot_frames, snapshot_success_flags, snapshot_labels = zip(
                *snapshot_items
            )
            snapshot_dir = os.path.join(self.output_dir, "rollout_snapshot")
            snapshot_path = os.path.join(
                snapshot_dir,
                rollout_snapshot_filename(epoch=epoch, env_seeds=self.env_seeds),
            )
            save_image_grid(
                list(snapshot_frames),
                snapshot_path,
                success_flags=list(snapshot_success_flags),
                labels=list(snapshot_labels),
            )

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            value = float(np.mean(value))
            log_data[name] = value
            if prefix == "test/":
                self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
                log_data[prefix + "max_score"] = self.max_rewards[prefix]

        if total_scores:
            total_mean = float(np.mean(total_scores))
            log_data["total/mean_score"] = total_mean
            self.max_rewards["total/"] = max(self.max_rewards["total/"], total_mean)
            log_data["total/max_score"] = self.max_rewards["total/"]

        return log_data

    def undo_transform_action(self, action, *, base_pos=None, base_rot=None):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            raise NotImplementedError("Dual-arm not supported yet.")
        # The last dimension is used for gripper non-binary state prediction, discard it during inference
        # this additional non-binary dimension is only used during training for enhancing action target variability while the gripper is open/closin
        if action.shape[-1] > 10:
            action = action[..., :-1]  # [B,T,Da-1]

        rot_dim = action.shape[-1] - 4  # pos(3) + gripper(1)
        pos = action[..., :3]
        rot = action[..., 3 : 3 + rot_dim]
        gripper = action[..., [-1]]
        if self.action_rep == "delta":
            B, T = pos.shape[:2]
            base_rot = base_rot.reshape(B, 1, 3, 3)
            rot_mat = self.delta_rotation_transformer.forward(rot.reshape(B * T, 6))
            rot_mat = rot_mat.reshape(B, T, 3, 3)
            # Convert chunked EE-frame deltas into world-frame absolute targets.
            rot = (base_rot @ rot_mat).reshape(B, T, 9)[..., :6]
            pos = base_pos[:, None, :] + (base_rot @ pos[..., None])[..., 0]

        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        return uaction

    def preprocess_obs(self, obs):
        if self.mirror_augmentation.enable:
            center_lowdim_observations(
                obs, self.pos_key, self.rot_key, self.mirror_augmentation
            )
        # not the first 2 cols, but the first 2 rows of the 3x3 matrix
        if self.obs_rot_is_6d:
            obs[self.rot_key] = obs[self.rot_key][..., :6]

        # add task embedding and robot index
        B, T = obs[self.pos_key].shape[:2]
        D = self.task_embedding.shape[0]

        task_embedding = np.broadcast_to(self.task_embedding, (B, T, D))
        task_embedding = task_embedding.astype(np.float32, copy=False)
        obs["task_embedding"] = task_embedding.copy()
        L, E = self.task_language_tokens.shape
        task_language_tokens = np.broadcast_to(
            self.task_language_tokens,
            (B, T, L, E),
        )
        task_language_tokens = task_language_tokens.astype(np.float32, copy=False)
        obs["task_language_tokens"] = task_language_tokens.copy()
        obs["robot_id"] = np.full((B, T, 1), self.robot_id, dtype=np.float32)

    def _current_oracle_focus_info(self, env, n_envs: int):
        if not self.enable_oracle_focus_info:
            return None
        infos = env.call("get_oracle_focus_info")
        if len(infos) != int(n_envs):
            raise RuntimeError(
                f"Expected {n_envs} oracle focus info entries, got {len(infos)}"
            )
        return self._stack_oracle_focus_infos(infos, repeat=self.n_obs_steps)

    def _oracle_focus_info_from_step_infos(self, infos):
        if not self.enable_oracle_focus_info:
            return None
        return self._stack_oracle_focus_infos(infos, repeat=None)

    def _stack_oracle_focus_infos(self, infos, repeat: Optional[int]):
        infos = list(infos)
        if not infos:
            return None
        keys = sorted(
            {
                key
                for info in infos
                if isinstance(info, dict)
                for key in info
                if str(key).startswith("oracle_target_")
            }
        )
        if not keys:
            return None

        out = {}
        for key in keys:
            values = []
            for info in infos:
                if key not in info:
                    return None
                arr = np.asarray(info[key])
                if repeat is not None:
                    arr = np.repeat(arr[None], int(repeat), axis=0)
                else:
                    arr = self._pad_oracle_temporal(arr)
                values.append(arr)
            out[key] = np.stack(values, axis=0)
        return out

    def _pad_oracle_temporal(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.shape[0] >= self.n_obs_steps:
            return arr[-self.n_obs_steps :]
        pad_count = int(self.n_obs_steps) - int(arr.shape[0])
        pad = np.repeat(arr[:1], pad_count, axis=0)
        return np.concatenate([pad, arr], axis=0)
