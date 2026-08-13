"""Playback Seeker dataset caches from stored frames or open-loop actions."""

from __future__ import annotations

import pathlib
from typing import List, Optional, Sequence

from ..dataset.cache import (
    is_cache_dir,
    load_metadata,
    load_numpy_array,
    resolve_cache_dir,
)


def _hdf5_candidates_from_dir(path: pathlib.Path) -> List[pathlib.Path]:
    """Return likely source HDF5 candidates for a dataset task directory."""
    exact = path / f"{path.name}.hdf5"
    candidates = [exact]
    candidates.extend(sorted(path.glob("*.hdf5")))

    out = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _existing_hdf5_candidates(path: pathlib.Path) -> List[pathlib.Path]:
    return [candidate for candidate in _hdf5_candidates_from_dir(path) if candidate.exists()]


def _source_hdf5_from_dir(dataset_path: pathlib.Path) -> pathlib.Path:
    candidates = _existing_hdf5_candidates(dataset_path)
    if not candidates and is_cache_dir(dataset_path):
        candidates = _existing_hdf5_candidates(dataset_path.parent)

    if len(candidates) == 1:
        return candidates[0]

    preferred = dataset_path / f"{dataset_path.name}.hdf5"
    if len(candidates) > 1 and preferred.exists():
        return preferred.resolve()

    if len(candidates) > 1:
        raise ValueError(
            "Multiple source HDF5 candidates found; pass the HDF5 "
            f"as --dataset-path with --cache-dir. Candidates: {candidates}"
        )

    raise FileNotFoundError(
        f"No source HDF5 found under dataset directory: {dataset_path}"
    )


def _cum_lengths(lengths: List[int]) -> np.ndarray:
    """Return cumulative episode lengths with a leading zero."""
    import numpy as np

    return np.cumsum([0] + [int(x) for x in lengths]).astype(np.int64)


def _open_lmdb_read(path: pathlib.Path):
    """Open LMDB in read-only mode."""
    import lmdb

    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=False,
        max_readers=2048,
    )


def _camera_name_from_rgb_key(key: str) -> str:
    """Map an observation image key to a robosuite camera name."""
    return key[: -len("_image")] if key.endswith("_image") else key


def _sorted_demo_keys(h5_file) -> List[str]:
    """Return robomimic demo keys sorted by integer suffix."""
    return sorted(h5_file["data"].keys(), key=lambda key: int(key.split("_")[-1]))


def _posmat_action_to_env_action(actions: np.ndarray) -> np.ndarray:
    """Convert ``[T, 3+9+g]`` actions to robosuite ``[T, 3+rotvec+g]``."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    if actions.ndim != 2 or actions.shape[1] < 12:
        raise ValueError(
            f"Expected pos+rotmat action shape (T, >=12), got {actions.shape}"
        )
    pos = actions[:, :3]
    rot = actions[:, 3:12].reshape(-1, 3, 3)
    gripper = actions[:, 12:]
    rotvec = R.from_matrix(rot).as_rotvec()
    actions = np.concatenate([pos, rotvec, gripper], axis=1).astype(np.float32)
    if actions.shape[1] > 7:
        actions = actions[:, :7]
    if actions.shape[1] != 7:
        raise ValueError(f"Expected single-arm action dim 7, got {actions.shape}")
    return actions


def _compose_horizontal(images: Sequence[np.ndarray]) -> np.ndarray:
    """Compose HWC BGR uint8 images left-to-right."""
    import cv2

    if len(images) == 0:
        raise ValueError("No images to compose")

    heights = [int(im.shape[0]) for im in images]
    target_h = max(heights)
    resized = []
    for im in images:
        if im.ndim != 3 or im.shape[-1] != 3:
            raise ValueError(f"Expected HWC image with 3 channels, got {im.shape}")
        if im.shape[0] != target_h:
            scale = target_h / float(im.shape[0])
            target_w = max(1, int(round(im.shape[1] * scale)))
            im = cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(im)
    return cv2.hconcat(resized)


def _obs_image_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an env RGB observation image into OpenCV BGR uint8."""
    import cv2
    import numpy as np

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class DatasetPlayback:
    """Playback cached observations or open-loop absolute actions."""

    def __init__(
        self,
        dataset_path: str,
        *,
        use_obs: bool,
        use_actions: bool,
        start_from: int = 0,
        cache_dir: Optional[str] = None,
        cameras: Optional[Sequence[str]] = None,
        wait_ms: int = 1,
        save_first_frame: Optional[str] = None,
        show_window: bool = True,
        max_frames: Optional[int] = None,
    ):
        if use_obs == use_actions:
            raise ValueError("Select exactly one playback mode: --use-obs or --use-actions")

        self.dataset_path = pathlib.Path(dataset_path).expanduser().resolve()
        self.cache_dir = resolve_cache_dir(dataset_path, cache_dir)
        self.meta, _ = load_metadata(str(self.cache_dir))
        self.episode_lengths = list(map(int, self.meta["episode_lengths"]))
        self.cum_lengths = _cum_lengths(self.episode_lengths)
        self.use_obs = bool(use_obs)
        self.use_actions = bool(use_actions)
        self.start_from = int(start_from)
        self.wait_ms = int(wait_ms)
        self.save_first_frame = save_first_frame
        self.show_window = bool(show_window)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.frames_shown = 0

        rgb_keys = list(self.meta.get("rgb_keys", []))
        if not rgb_keys:
            raise ValueError(f"Cache metadata has no rgb_keys: {self.cache_dir}")
        if cameras is None or len(cameras) == 0:
            self.rgb_keys = rgb_keys
        else:
            missing = [key for key in cameras if key not in rgb_keys]
            if missing:
                raise ValueError(
                    f"Requested cameras not in cache rgb_keys: {missing}. "
                    f"Available: {rgb_keys}"
                )
            self.rgb_keys = list(cameras)

        self.lmdb_env = None
        self.env = None
        self.h5_file = None
        self.demo_keys: List[str] = []
        self.actions = None

        if self.use_obs:
            self.lmdb_env = _open_lmdb_read(self.cache_dir / "images.lmdb")
        else:
            self._init_action_playback()

    def _init_action_playback(self) -> None:
        """Build an absolute-controller env and load cache absolute actions."""
        import h5py
        import mimicgen  # noqa: F401
        import numpy as np
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils
        from robomimic.config import config_factory

        from ..env.robomimic import update_env_controller

        source_hdf5_path = self._resolve_source_hdf5_path()
        self.actions = load_numpy_array(
            str(self.cache_dir),
            "action/absolute_action.npy",
        )
        if self.actions.shape[0] != int(self.cum_lengths[-1]):
            raise ValueError(
                f"absolute_action length mismatch: {self.actions.shape[0]} actions "
                f"for {int(self.cum_lengths[-1])} cached frames"
            )

        config = config_factory(algo_name="bc")
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(str(source_hdf5_path))
        if self.meta.get("env_name"):
            env_meta["env_name"] = self.meta["env_name"]
        env_meta = update_env_controller(env_meta, action_rep="absolute")

        camera_names = [_camera_name_from_rgb_key(key) for key in self.rgb_keys]
        image_size = int(self.meta.get("image_size", 256))
        self.env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=camera_names,
            camera_height=image_size,
            camera_width=image_size,
            reward_shaping=False,
        )

        self.h5_file = h5py.File(source_hdf5_path, "r")
        self.demo_keys = _sorted_demo_keys(self.h5_file)

    def _resolve_source_hdf5_path(self) -> pathlib.Path:
        """Resolve the raw HDF5 file needed for action playback initial states."""
        source_hdf5 = self.meta.get("source_hdf5")
        if source_hdf5 is not None:
            path = pathlib.Path(source_hdf5).expanduser().resolve()
        elif self.dataset_path.is_file():
            path = self.dataset_path
        elif self.dataset_path.is_dir():
            path = _source_hdf5_from_dir(self.dataset_path)
        else:
            raise ValueError(
                "--use-actions needs the source HDF5 for episode initial states. "
                "Use a cache with source_hdf5 metadata or pass the HDF5 as "
                "--dataset-path with --cache-dir."
            )
        if not path.exists():
            raise FileNotFoundError(f"Source HDF5 not found: {path}")
        return path

    def close(self):
        """Close opened resources."""
        if getattr(self, "lmdb_env", None) is not None:
            self.lmdb_env.close()
            self.lmdb_env = None
        if getattr(self, "h5_file", None) is not None:
            self.h5_file.close()
            self.h5_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __len__(self):
        return len(self.episode_lengths)

    def playback(self):
        """Run selected playback mode."""
        if self.start_from < 0 or self.start_from >= len(self):
            raise ValueError(
                f"--start-from {self.start_from} out of range for {len(self)} episodes"
            )
        if self.use_obs:
            self._playback_obs()
        else:
            self._playback_actions()

    def _decode_cached_frame(self, txn, rgb_key: str, global_idx: int) -> np.ndarray:
        """Decode one cached frame as OpenCV BGR uint8 image."""
        from ..util.image_io import decode_jpg_bytes

        key = f"{rgb_key}/{int(global_idx):08d}".encode("ascii")
        buf = txn.get(key)
        if buf is None:
            raise KeyError(f"Missing LMDB key: {key!r}")
        # The cache decode path returns RGB tensors for training by default.
        # Convert to BGR here because OpenCV display/write expects BGR.
        return decode_jpg_bytes(buf, bgr_to_rgb=True, to_float=False, fmt="HWC")

    def _show_frame(self, image: np.ndarray, ep_idx: int, local_idx: int) -> bool:
        """Show and optionally save a composed playback frame."""
        import cv2

        if self.save_first_frame and ep_idx == self.start_from and local_idx == 0:
            cv2.imwrite(self.save_first_frame, image)
        if self.show_window:
            cv2.imshow("dataset_playback", image)
            cv2.waitKey(self.wait_ms)
        self.frames_shown += 1
        return self.max_frames is None or self.frames_shown < self.max_frames

    def _playback_obs(self) -> None:
        """Playback recorded cache images only."""
        from tqdm import tqdm

        with self.lmdb_env.begin(write=False) as txn:
            for ep_idx in tqdm(range(self.start_from, len(self)), desc="Playback obs"):
                step_lo = int(self.cum_lengths[ep_idx])
                step_hi = int(self.cum_lengths[ep_idx + 1])
                for local_idx, global_idx in enumerate(range(step_lo, step_hi)):
                    images = [
                        self._decode_cached_frame(txn, key, global_idx)
                        for key in self.rgb_keys
                    ]
                    keep_going = self._show_frame(
                        _compose_horizontal(images),
                        ep_idx=ep_idx,
                        local_idx=local_idx,
                    )
                    if not keep_going:
                        return

    def _source_demo_key(self, ep_idx: int) -> str:
        """Map cache episode index to source HDF5 demo key."""
        if "source_demo_indices" in self.meta:
            source_idx = int(self.meta["source_demo_indices"][ep_idx])
        else:
            source_idx = ep_idx
            if ep_idx == self.start_from:
                print(
                    "[playback] cache has no source_demo_indices; assuming cache "
                    "episodes map to source HDF5 demos by the same index."
                )
        if source_idx < 0 or source_idx >= len(self.demo_keys):
            raise IndexError(
                f"Source demo index {source_idx} out of range for "
                f"{len(self.demo_keys)} demos"
            )
        return self.demo_keys[source_idx]

    def _table_texture_file_for_episode(self, ep_idx: int) -> Optional[pathlib.Path]:
        """Return the texture file recorded for a cache episode, if available."""
        texture_meta = self.meta.get("table_texture")
        if not isinstance(texture_meta, dict):
            return None
        episode_files = texture_meta.get("episode_files")
        texture_dir = texture_meta.get("texture_dir")
        if episode_files is None or texture_dir is None:
            return None
        if ep_idx < 0 or ep_idx >= len(episode_files):
            raise IndexError(
                f"Texture metadata has {len(episode_files)} episodes, "
                f"requested {ep_idx}"
            )
        return pathlib.Path(texture_dir).expanduser().resolve() / str(
            episode_files[ep_idx]
        )

    def _playback_actions(self) -> None:
        """Open-loop playback by stepping cached absolute actions."""
        from tqdm import tqdm

        for ep_idx in tqdm(range(self.start_from, len(self)), desc="Playback actions"):
            step_lo = int(self.cum_lengths[ep_idx])
            step_hi = int(self.cum_lengths[ep_idx + 1])
            actions = _posmat_action_to_env_action(self.actions[step_lo:step_hi])

            demo_key = self._source_demo_key(ep_idx)
            demo = self.h5_file[f"data/{demo_key}"]
            initial_state = {
                "states": demo["states"][0],
                "model": demo.attrs["model_file"],
            }
            texture_file = self._table_texture_file_for_episode(ep_idx)
            if texture_file is not None:
                from ..env.mjcf_texture import apply_table_texture

                initial_state["model"] = apply_table_texture(
                    initial_state["model"],
                    texture_file=str(texture_file),
                )
            self.env.reset()
            self.env.reset_to(initial_state)

            for local_idx, action in enumerate(actions):
                obs = self.env.get_observation()
                images = []
                for key in self.rgb_keys:
                    if key not in obs:
                        raise KeyError(
                            f"Env observation missing image key {key!r}; "
                            f"available keys include {list(obs.keys())[:10]}"
                        )
                    images.append(_obs_image_to_bgr(obs[key]))
                keep_going = self._show_frame(
                    _compose_horizontal(images),
                    ep_idx=ep_idx,
                    local_idx=local_idx,
                )
                if not keep_going:
                    return
                self.env.step(action)


def main():
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help=(
            "Path to a rerendered cache directory. If a raw HDF5 path is given, "
            "the default sibling cache path is inferred."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--use-obs",
        action="store_true",
        help="Playback recorded cache images directly.",
    )
    mode.add_argument(
        "--use-actions",
        action="store_true",
        help="Open-loop rerender with cache action/absolute_action.npy.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Explicit rerendered cache directory.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help=(
            "RGB keys to compose horizontally. Defaults to every cache rgb_key, "
            "for example agentview_image robot0_eye_in_hand_image."
        ),
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=1,
        help="OpenCV wait time between frames.",
    )
    parser.add_argument(
        "--save-first-frame",
        type=str,
        default=None,
        help="Optional path for the first composed frame.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open an OpenCV window; useful with --save-first-frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames.",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start playback from this cache episode index.",
    )
    args = parser.parse_args()

    with DatasetPlayback(
        dataset_path=args.dataset_path,
        use_obs=args.use_obs,
        use_actions=args.use_actions,
        start_from=args.start_from,
        cache_dir=args.cache_dir,
        cameras=args.cameras,
        wait_ms=args.wait_ms,
        save_first_frame=args.save_first_frame or None,
        show_window=not args.no_window,
        max_frames=args.max_frames,
    ) as playback:
        playback.playback()


if __name__ == "__main__":
    main()
