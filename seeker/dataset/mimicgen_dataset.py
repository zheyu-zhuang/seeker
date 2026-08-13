"""MimicGen dataset backed by LMDB image cache + NumPy low-dimensional arrays."""

import os

import lmdb
import numpy as np
import torch

from seeker.model.common.normalizer import (
    LinearNormalizer,
    MultiRobotLinearNormalizer,
    SingleFieldLinearNormalizer,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from seeker.dataset.action import load_action_array, validate_action_rep
from seeker.dataset.cache import (
    check_cache,
    get_obs_keys,
    load_lowdim,
    load_metadata,
    load_numpy_array,
    load_task_instructions,
    resolve_cache_dir,
)
from seeker.dataset.oracle_cache import load_oracle_info
from seeker.util.image_io import decode_jpg_bytes
from seeker.util.formatting import find_all_keys, is_matched_key
from seeker.util.mirror_augmentation import (
    MirrorAugmentationConfig,
    action_world_to_mirror_frame,
    pose_world_to_mirror_frame,
)


class MimicGenDataset(torch.utils.data.Dataset):
    """Torch dataset for Seeker training/evaluation on cached MimicGen trajectories."""

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        image_size,
        horizon=16,
        val_ratio=0.02,
        n_demo=None,
        demo_count_mode="total",
        n_obs_steps=1,
        action_rep="absolute",
        lmdb_readahead=False,
        cache_dir=None,
        include_oracle_info=False,
        mirror_augmentation=None,
    ):
        self.n_obs_steps = int(n_obs_steps)
        if self.n_obs_steps < 1:
            raise ValueError("n_obs_steps must be >= 1")

        self.image_size = image_size
        self.horizon = int(horizon)
        self.val_ratio = float(val_ratio)
        self.action_rep = validate_action_rep(action_rep)
        self.mirror_augmentation = MirrorAugmentationConfig.from_config(
            mirror_augmentation
        )
        self.lmdb_readahead = bool(lmdb_readahead)
        self.include_oracle_info = bool(include_oracle_info)
        self.demo_count_mode = str(demo_count_mode).strip().lower()
        if self.demo_count_mode not in ("total", "per_task"):
            raise ValueError(
                "demo_count_mode must be one of ['total', 'per_task'], "
                f"got {demo_count_mode!r}"
            )
        self.rgb_keys, self.lowdim_keys = get_obs_keys(shape_meta)
        self.action_dim = shape_meta["action"]["shape"][0]
        self.pos_key = find_all_keys(
            "eef_pos", self.lowdim_keys, only_one_match=True, reject=["delta", "qpos"]
        )
        self.rot_key = find_all_keys(
            "eef_rot", self.lowdim_keys, only_one_match=True, reject=["delta"]
        )

        self.cache_dir = str(resolve_cache_dir(dataset_path, cache_dir))
        self.lmdb_path = os.path.join(self.cache_dir, "images.lmdb")
        self._lmdb_env = None
        self._lmdb_txn = None

        check_cache(cache_dir=self.cache_dir, dataset_path=dataset_path)
        self._load_cache_arrays()
        self.oracle_info = (
            load_oracle_info(self.cache_dir) if self.include_oracle_info else None
        )

        self.set_active_demos(n_demo)
        self.mode_set = False
        self.start_idx = 0
        self.end_idx = 0

    def set_active_demos(self, n_demo=None):
        """Select active episodes and recompute split/sample bookkeeping."""
        if n_demo is None:
            selected_episode_indices = np.arange(self.n_demo_all, dtype=np.int64)
        elif self.demo_count_mode == "per_task":
            n = int(n_demo)
            if n < 1:
                raise ValueError(f"n_demo must be >= 1, got {n}")
            task_groups = self._get_task_episode_groups()
            selected_episode_indices = []
            for group in task_groups:
                if len(group) < n:
                    raise ValueError(
                        "n_demo exceeds demos available for one task group: "
                        f"requested {n}, got {len(group)}"
                    )
                selected_episode_indices.extend(group[:n].tolist())
            selected_episode_indices = np.asarray(selected_episode_indices, dtype=np.int64)
        else:
            n = int(n_demo)
            if n < 1 or n > self.n_demo_all:
                raise ValueError(f"n_demo must be in [1, {self.n_demo_all}], got {n}")
            selected_episode_indices = np.arange(n, dtype=np.int64)

        self.episode_indices_active = np.asarray(selected_episode_indices, dtype=np.int64)
        self.n_demo_active = int(len(self.episode_indices_active))
        if self.n_demo_active < 1:
            raise ValueError("Active demo selection is empty")

        lens_all = np.asarray(self.episode_lengths_all, dtype=np.int64)
        self.episode_lengths_active = lens_all[self.episode_indices_active].astype(int).tolist()
        self.cum_lengths_active = np.cumsum([0] + self.episode_lengths_active).astype(
            np.int64
        )
        step_ranges = [
            np.arange(
                self.cum_lengths_all[ep_idx],
                self.cum_lengths_all[ep_idx + 1],
                dtype=np.int64,
            )
            for ep_idx in self.episode_indices_active.tolist()
        ]
        self.active_step_indices = np.concatenate(step_ranges, axis=0)
        self.n_samples_active = int(self.active_step_indices.shape[0])

        self.task_embedding_episode = self.task_embedding_episode_all[
            self.episode_indices_active
        ].astype(np.float32, copy=False)
        self.task_language_tokens_episode = (
            None
            if self.task_language_tokens_episode_all is None
            else self.task_language_tokens_episode_all[
                self.episode_indices_active
            ].astype(np.float32, copy=False)
        )
        self.task_id_episode = self.task_id_episode_all[
            self.episode_indices_active
        ].astype(np.int64, copy=False)
        self.robot_id_episode = self.robot_id_episode_all[self.episode_indices_active].astype(
            np.int64,
            copy=False,
        )
        if self.task_instructions_all is not None:
            self.task_instructions = [
                self.task_instructions_all[int(ep_idx)]
                for ep_idx in self.episode_indices_active.tolist()
            ]
        else:
            self.task_instructions = None

        self.n_train_episodes = int(self.n_demo_active * (1.0 - self.val_ratio))
        self.n_eval_episodes = self.n_demo_active - self.n_train_episodes
        self.train_length = int(self.cum_lengths_active[self.n_train_episodes])
        self.eval_length = int(self.n_samples_active - self.train_length)

    def set_mode(self, mode: str):
        """Select sampling range: ``train``, ``eval``, or ``all``."""
        ranges = {
            "train": (0, self.train_length),
            "eval": (self.train_length, self.train_length + self.eval_length),
            "all": (0, self.n_samples_active),
        }
        if mode not in ranges:
            raise ValueError(
                f"mode must be one of ['train','eval','all'], got {mode!r}"
            )
        self.start_idx, self.end_idx = map(int, ranges[mode])
        self.mode_set = True

    def __len__(self):
        return int(self.end_idx - self.start_idx)

    def __getitem__(self, idx):
        if not self.mode_set:
            raise RuntimeError("Dataset mode not set. set_mode('mode').")
        obs_idx = int(self.start_idx + idx)
        eps_index, obs_indices, action_indices = self.sampler(obs_idx)
        return self._make_sample(
            eps_index=eps_index,
            obs=self.get_obs(eps_index, obs_indices),
            action=self._get_action_window(obs_idx=obs_idx, action_indices=action_indices),
            obs_indices=obs_indices,
        )

    def get_normalizer(self) -> MultiRobotLinearNormalizer:
        """Build one per-robot normalizer over the active dataset slice."""
        active_action = self.action[self.active_step_indices]
        active_lowdim = {
            key: self.lowdim[key][self.active_step_indices].astype(np.float32, copy=False)
            for key in self.lowdim_keys
        }

        action_dim = int(self.action_dim)
        if action_dim < 9:
            raise ValueError(
                "action_dim must include pos(3)+rot6d(6) to build the action "
                f"normalizer, got {action_dim}"
            )
        if self.mirror_augmentation.enable:
            active_lowdim = dict(active_lowdim)
            active_lowdim[self.pos_key], active_lowdim[self.rot_key] = (
                pose_world_to_mirror_frame(
                    active_lowdim[self.pos_key],
                    active_lowdim[self.rot_key],
                    self.mirror_augmentation,
                )
            )
            active_action = action_world_to_mirror_frame(
                active_action,
                self.mirror_augmentation,
                action_rep=self.action_rep,
                action_dim=action_dim,
            )

        step_robot_ids = np.repeat(
            np.asarray(self.robot_id_episode, dtype=np.int64).reshape(-1),
            np.asarray(self.episode_lengths_active, dtype=np.int64),
        )

        img_norm = get_image_range_normalizer()
        task_embedding_norm = get_identity_normalizer_from_stat(
            array_to_stats(self.task_embedding_episode)
        )
        robot_id_norm = get_identity_normalizer_from_stat(
            array_to_stats(self.robot_id_episode.reshape(-1, 1).astype(np.float32))
        )

        normalizers = {}
        for robot_id in np.unique(step_robot_ids).tolist():
            robot_id = int(robot_id)
            mask = step_robot_ids == robot_id
            norm = LinearNormalizer()

            stat = array_to_stats(active_action[mask])
            width = int(stat["min"].shape[0])
            if width % action_dim != 0:
                raise ValueError(
                    "action stats width must be divisible by per-step action_dim: "
                    f"width={width} action_dim={action_dim}"
                )

            if self.mirror_augmentation.enable:
                self._symmetrize_stat_x_dims(stat, width=width, stride=action_dim)

            input_min = stat["min"]
            input_range = stat["max"] - input_min
            ignore_dim = input_range < 1e-7
            input_range[ignore_dim] = 2

            scale = 2 / input_range
            offset = -1 - scale * input_min
            offset[ignore_dim] = -input_min[ignore_dim]

            rot_mask = np.zeros(width, dtype=bool)
            for start in range(0, width, action_dim):
                rot_mask[start + 3 : start + 9] = True
            scale[rot_mask] = 1
            offset[rot_mask] = 0

            norm["action"] = SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict=stat,
            )

            for key in self.lowdim_keys:
                stat = array_to_stats(active_lowdim[key][mask])
                if is_matched_key("pos", key):
                    if self.mirror_augmentation.enable:
                        self._symmetrize_stat_x_dims(
                            stat, width=stat["min"].shape[0], stride=3
                        )
                    norm[key] = get_range_normalizer_from_stat(stat)
                elif is_matched_key("qpos", key):
                    norm[key] = get_range_normalizer_from_stat(stat)
                elif is_matched_key("rot", key):
                    norm[key] = get_identity_normalizer_from_stat(stat)
                else:
                    raise RuntimeError(f"unsupported lowdim key: {key}")

            for key in self.rgb_keys:
                norm[key] = img_norm

            norm["task_embedding"] = task_embedding_norm
            norm["robot_id"] = robot_id_norm
            normalizers[robot_id] = norm

        return MultiRobotLinearNormalizer(normalizers)

    @staticmethod
    def _symmetrize_stat_x_dims(stat: dict, *, width: int, stride: int) -> None:
        """Make x-position dimensions symmetric around zero in normalizer stats."""
        for idx in range(0, int(width), int(stride)):
            abs_max = max(abs(float(stat["min"][idx])), abs(float(stat["max"][idx])))
            stat["min"][idx] = -abs_max
            stat["max"][idx] = abs_max

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_lmdb_env"] = None
        d["_lmdb_txn"] = None
        return d

    def _get_lmdb_env(self):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=self.lmdb_readahead,
                meminit=False,
                subdir=False,
                max_readers=2048,
            )
        return self._lmdb_env

    def _get_lmdb_txn(self):
        if self._lmdb_txn is None:
            self._lmdb_txn = self._get_lmdb_env().begin(write=False, buffers=True)
        return self._lmdb_txn

    def active_to_global_indices(self, active_indices) -> np.ndarray:
        active_indices = np.asarray(active_indices, dtype=np.int64)
        return self.active_step_indices[active_indices]

    def episode_active_indices(self, episode_idx: int) -> np.ndarray:
        episode_start = int(self.cum_lengths_active[episode_idx])
        ep_len = int(self.episode_lengths_active[episode_idx])
        return np.arange(episode_start, episode_start + ep_len, dtype=np.int64)

    def episode_global_indices(self, episode_idx: int) -> np.ndarray:
        return self.active_to_global_indices(self.episode_active_indices(episode_idx))

    def optional_episode_lowdim(self, episode_idx: int, key: str):
        global_indices = self.episode_global_indices(episode_idx)
        if key in self.lowdim:
            return self.lowdim[key][global_indices].astype(np.float32, copy=False), key
        try:
            arr = load_numpy_array(self.cache_dir, os.path.join("lowdim", f"{key}.npy"))
        except FileNotFoundError:
            return None, None
        return arr[global_indices].astype(np.float32, copy=False), key

    def task_sample_ranges(self):
        if not self.mode_set:
            raise RuntimeError("Dataset mode not set. set_mode('mode').")
        task_to_ranges = {}
        for ep_idx, task_id in enumerate(self.task_id_episode.tolist()):
            ep_start = int(self.cum_lengths_active[ep_idx])
            ep_end = int(self.cum_lengths_active[ep_idx + 1])
            lo = max(ep_start, int(self.start_idx))
            hi = min(ep_end, int(self.end_idx))
            if hi > lo:
                task_to_ranges.setdefault(int(task_id), []).append(
                    (lo - int(self.start_idx), hi - int(self.start_idx))
                )
        return task_to_ranges

    def get_obs(self, eps_index, obs_indices):
        """Fetch RGB and lowdim observations for a list of active frame indices."""
        active_obs_indices = np.asarray(obs_indices, dtype=np.int64)
        global_obs_indices = self.active_to_global_indices(active_obs_indices)
        obs = {}
        txn = self._get_lmdb_txn()
        for img_key in self.rgb_keys:
            obs[img_key] = np.stack(
                [
                    self._decode_lmdb_image(txn, img_key, gidx)
                    for gidx in global_obs_indices.tolist()
                ],
                axis=0,
            )

        obs.update(
            {
                key: self.lowdim[key][global_obs_indices].astype(np.float32, copy=False)
                for key in self.lowdim_keys
            }
        )

        n = len(obs_indices)
        task_emb = self.task_embedding_episode[eps_index].astype(np.float32, copy=False)
        obs["task_embedding"] = np.repeat(task_emb[None, :], n, axis=0)
        if self.task_language_tokens_episode is not None:
            lang_tokens = self.task_language_tokens_episode[eps_index].astype(
                np.float32,
                copy=False,
            )
            obs["task_language_tokens"] = np.repeat(
                lang_tokens[None, :, :],
                n,
                axis=0,
            )
        obs["task_id"] = np.full(
            (n, 1), self.task_id_episode[eps_index], dtype=np.int64
        )
        obs["robot_id"] = np.full(
            (n, 1), self.robot_id_episode[eps_index], dtype=np.float32
        )
        return obs

    def _decode_lmdb_image(self, txn, img_key: str, gidx: int) -> np.ndarray:
        key = f"{img_key}/{int(gidx):08d}".encode("ascii")
        buf = txn.get(key)
        if buf is None:
            raise KeyError(f"Missing LMDB key: {key!r}")
        return decode_jpg_bytes(
            buf, image_size=self.image_size, to_float=False, fmt="CHW"
        )

    def sampler(self, idx: int):
        """Map global sample index to episode id, obs indices, and action indices."""
        total = int(self.cum_lengths_active[-1])
        if idx < 0 or idx >= total:
            raise IndexError(f"idx {idx} out of range [0, {total})")

        episode_idx = int(np.searchsorted(self.cum_lengths_active, idx, side="right") - 1)
        episode_start = int(self.cum_lengths_active[episode_idx])
        ep_len = int(self.episode_lengths_active[episode_idx])
        local_idx = int(idx - episode_start)

        obs_indices = (
            episode_start
            + np.maximum(local_idx + np.arange(-(self.n_obs_steps - 1), 1), 0)
        ).tolist()
        action_indices = (
            episode_start
            + np.minimum(local_idx + np.arange(self.horizon), ep_len - 1)
        ).tolist()
        return episode_idx, obs_indices, action_indices

    def get_trajectory(self, episode_idx: int):
        """Return the full trajectory for one active episode."""
        assert 0 <= episode_idx < self.n_demo_active, "Invalid episode index."
        episode_start = int(self.cum_lengths_active[episode_idx])
        ep_len = int(self.episode_lengths_active[episode_idx])
        obs_indices = list(range(episode_start, episode_start + ep_len))
        global_obs_indices = self.active_to_global_indices(obs_indices)
        return self._make_sample(
            eps_index=episode_idx,
            obs=self.get_obs(episode_idx, obs_indices),
            action=self.action[global_obs_indices],
            obs_indices=obs_indices,
        )

    def _make_sample(self, eps_index, obs, action, obs_indices):
        sample = {
            "obs": obs,
            "action": action.astype(np.float32, copy=False),
            "obs_index": np.array(obs_indices, dtype=np.int32),
        }
        if self.task_instructions is not None:
            sample["task_instruction"] = self.task_instructions[eps_index]
        if self.oracle_info is not None:
            sample["oracle_info"] = self._get_oracle_info(obs_indices)
        return sample

    def _get_oracle_info(self, obs_indices) -> dict[str, np.ndarray]:
        active_obs_indices = np.asarray(obs_indices, dtype=np.int64)
        global_obs_indices = self.active_to_global_indices(active_obs_indices)
        return self._slice_oracle_info(self.oracle_info, global_obs_indices)

    def _slice_oracle_info(self, value, indices):
        if isinstance(value, dict):
            out = {}
            for key, subvalue in value.items():
                sliced = self._slice_oracle_info(subvalue, indices)
                if sliced is not None:
                    out[key] = sliced
            return out
        out = np.asarray(value)[indices]
        if out.dtype.kind in ("O", "U", "S"):
            return None
        return out.astype(np.float32, copy=False)

    def _get_action_window(self, obs_idx: int, action_indices) -> np.ndarray:
        if self.action_rep == "delta":
            action = self.action[int(self.active_to_global_indices([obs_idx])[0])]
        else:
            global_action_indices = self.active_to_global_indices(action_indices)
            action = self.action[global_action_indices]
        return action.reshape(self.horizon, -1)

    def _load_cache_arrays(self) -> None:
        """Load cache metadata, actions, and low-dimensional arrays."""
        self.meta, self.episode_lengths_all = load_metadata(self.cache_dir)
        self.n_demo_all = len(self.episode_lengths_all)
        self.cum_lengths_all = np.cumsum([0] + self.episode_lengths_all).astype(np.int64)
        meta = dict(self.meta)
        meta.setdefault("lowdim_keys", list(self.lowdim_keys))
        self.action = load_action_array(
            cache_dir=self.cache_dir,
            meta=meta,
            horizon=self.horizon,
            action_dim=self.action_dim,
            action_rep=self.action_rep,
        )
        self.lowdim = load_lowdim(self.cache_dir, self.lowdim_keys)
        self.task_embedding_episode_all = load_numpy_array(
            self.cache_dir, os.path.join("lowdim", "task_embedding.npy")
        ).astype(np.float32, copy=False)
        try:
            self.task_language_tokens_episode_all = load_numpy_array(
                self.cache_dir, os.path.join("lowdim", "task_language_tokens.npy")
            ).astype(np.float32, copy=False)
        except FileNotFoundError:
            self.task_language_tokens_episode_all = None
        try:
            self.task_id_episode_all = np.asarray(
                load_numpy_array(self.cache_dir, os.path.join("lowdim", "task_id.npy"))
            ).reshape(-1).astype(np.int64, copy=False)
        except FileNotFoundError:
            self.task_id_episode_all = np.zeros(
                (self.n_demo_all,), dtype=np.int64
            )
        self.robot_id_episode_all = np.asarray(
            load_numpy_array(self.cache_dir, os.path.join("lowdim", "robot_id.npy"))
        ).reshape(-1).astype(np.int64, copy=False)
        self.task_instructions_all = load_task_instructions(self.cache_dir)

    def _get_task_episode_groups(self):
        """Return contiguous episode groups that define one task each."""
        source_counts = self.meta.get("source_episode_counts")
        if not isinstance(source_counts, list) or len(source_counts) == 0:
            raise ValueError(
                "Merged cache is missing required source_episode_counts metadata. "
                "Rebuild the merged cache with seeker merge-datasets."
            )
        counts = [int(x) for x in source_counts]
        if sum(counts) != self.n_demo_all:
            raise ValueError(
                "Invalid source_episode_counts in cache metadata: "
                f"sum={sum(counts)} != n_demo_all={self.n_demo_all}"
            )
        groups = []
        start = 0
        for count in counts:
            if count < 1:
                raise ValueError(
                    "Invalid source_episode_counts in cache metadata: "
                    f"got non-positive count {count}"
                )
            stop = start + count
            groups.append(np.arange(start, stop, dtype=np.int64))
            start = stop
        return groups
