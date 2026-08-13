"""Rerender robomimic/mimicgen HDF5 demos into LMDB caches."""

from __future__ import annotations

import os
import json
import shutil
import argparse
import queue
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import h5py
import lmdb
import numpy as np
import cv2
from tqdm import tqdm

import mimicgen  # noqa: F401
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from .convert_dataset_actions import convert_actions
from ..dataset.cache import archive_key, archive_path, default_cache_dir_for_dataset
from ..dataset.oracle_cache import (
    OracleContext,
    OracleFrameCollector,
    build_oracle_arrays,
    build_oracle_context,
)
from ..dataset.action import absolute_posmat_to_delta_chunks, action_to_posmat
from ..env.robomimic import (
    extract_trajectory,
    require_upstream_controller_config,
    update_env_controller,
)
from ..util.task_meta import (
    setup_task_embedding_cache,
    env_name_to_meta,
)
from ..util.image_io import encode_rgb_to_jpg_bytes
from ..env.mjcf_texture import (
    apply_table_texture,
    list_texture_files,
)
from seeker import TEXTURES_DIR

COMMIT_EVERY_DEFAULT = 5000
LMDB_MAP_SIZE_GB_DEFAULT = 32
JPEG_QUALITY_DEFAULT = 90
PROGRESS_EVENT_DEMO_DONE = "demo_done"
PROGRESS_POLL_INTERVAL_SEC = 0.2


def _remove_existing_output_dir(path: str | Path, *, overwrite: bool) -> None:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return
    if not overwrite:
        resp = input(f"Output dir exists: {path}\nOverwrite? (y/n): ").strip().lower()
        if resp != "y":
            raise SystemExit("Canceled.")
    shutil.rmtree(path)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _camera_obs_keys(camera_names: List[str]) -> List[str]:
    # Normalize to robosuite obs key convention: "<camera>_image"
    out = []
    for name in camera_names:
        if name.endswith("_image"):
            out.append(name)
        else:
            out.append(f"{name}_image")
    return out


def _sorted_demo_keys(h5_file: h5py.File) -> List[str]:
    return sorted(h5_file["data"].keys(), key=lambda x: int(x[5:]))


def _source_demo_indices(
    dataset_path: str,
    start_index: int,
    n_demo: Optional[int],
) -> List[int]:
    """Return source demo indices selected for rerender."""
    with h5py.File(str(Path(dataset_path).expanduser().resolve()), "r") as h5_file:
        demos = _sorted_demo_keys(h5_file)
    start_index = int(start_index)
    if start_index < 0 or start_index >= len(demos):
        raise ValueError(
            f"start episode {start_index} out of range (episodes={len(demos)})"
        )
    selected = [int(ep[5:]) for ep in demos[start_index:]]
    if n_demo is not None:
        selected = selected[: int(n_demo)]
    if not selected:
        raise ValueError("No demos selected for rerender")
    return selected


def _sibling_cache_dir_with_suffix(cache_dir: str | Path, suffix: str) -> str:
    cache_path = Path(cache_dir).expanduser().resolve()
    if not suffix:
        return str(cache_path)
    return str(cache_path.parent / f"{cache_path.name}{suffix}")


def _discover_task_datasets(
    datasets_root: str,
    tasks: Optional[List[str]] = None,
) -> List[tuple[str, str]]:
    """Discover raw task HDF5 files under a MimicGen-style datasets root."""
    root = Path(datasets_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"datasets_root not found: {root}")

    if tasks:
        task_names = [str(task) for task in tasks]
    else:
        task_names = sorted(
            child.name
            for child in root.iterdir()
            if child.is_dir() and (child / f"{child.name}.hdf5").is_file()
        )

    if not task_names:
        raise ValueError(f"No task datasets found under {root}")

    datasets = []
    missing = []
    for task_name in task_names:
        hdf5_path = root / task_name / f"{task_name}.hdf5"
        if hdf5_path.is_file():
            datasets.append((task_name, str(hdf5_path)))
        else:
            missing.append(str(hdf5_path))
    if missing:
        raise FileNotFoundError(
            "Missing task HDF5 files:\n" + "\n".join(f"  {path}" for path in missing)
        )
    return datasets


def _split_contiguous(values: List[int], num_chunks: int) -> List[List[int]]:
    """Split values into contiguous non-empty chunks."""
    num_chunks = max(1, min(int(num_chunks), len(values)))
    base = len(values) // num_chunks
    rem = len(values) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        size = base + (1 if i < rem else 0)
        chunks.append(values[start : start + size])
        start += size
    return [chunk for chunk in chunks if chunk]


def _drain_progress_events(
    progress_queue,
    pbar,
    samples: int,
    demos: int,
) -> tuple[int, int, bool]:
    drained = False
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            break
        drained = True
        if event.get("event") != PROGRESS_EVENT_DEMO_DONE:
            continue
        samples += int(event["n_samples"])
        demos += 1
        pbar.update(1)
        pbar.set_postfix_str(f"samples={samples}")
    return samples, demos, drained


class DatasetRerenderToCache:
    def __init__(
        self,
        dataset_path: str,
        out_cache_dir: str,
        camera_resolution: int,
        table_texture_every: int = None,
        texture_rank_by_source_demo: Optional[Dict[int, int]] = None,
        oracle_camera: Optional[str] = "agentview",
        oracle_patch_size: int = 16,
        oracle_min_patch_area_fraction: float = 0.05,
        oracle_min_mask_pixels: int = 16,
        overwrite: bool = False,
    ):
        self.input_path = str(Path(dataset_path).expanduser().resolve())
        self.out_dir = str(Path(out_cache_dir).expanduser().resolve())
        self.res = int(camera_resolution)
        self.oracle_camera = None if oracle_camera is None else str(oracle_camera)
        self.oracle_patch_size = int(oracle_patch_size)
        self.oracle_min_patch_area_fraction = float(oracle_min_patch_area_fraction)
        self.oracle_min_mask_pixels = int(oracle_min_mask_pixels)
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input dataset not found: {self.input_path}")

        _remove_existing_output_dir(self.out_dir, overwrite=overwrite)
        os.makedirs(self.out_dir, exist_ok=True)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=self.input_path)

        self.source_env_meta = require_upstream_controller_config(env_meta)
        self.env_meta = update_env_controller(env_meta, action_rep="absolute")

        # Source-controller env converts raw data/<demo>/actions into absolute
        # targets before the rendering env executes them.
        self.action_conversion_env = EnvUtils.create_env_from_metadata(
            env_meta=self.source_env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )

        # Create a rendering env with the absolute controller.
        self.env = EnvUtils.create_env_for_data_processing(
            env_meta=self.env_meta,
            camera_names=self.env_meta["env_kwargs"]["camera_names"],
            camera_height=self.res,
            camera_width=self.res,
            reward_shaping=False,
        )

        self.camera_names = list(self.env_meta["env_kwargs"]["camera_names"])
        self.rgb_keys = _camera_obs_keys(self.camera_names)
        self.oracle = self._init_oracle_context()
        self.rerender_counter = 0
        self.table_texture_every = table_texture_every
        self.table_texture_files = None
        if self.table_texture_every is not None:
            texture_dir = Path(TEXTURES_DIR) / "train"
            self.table_texture_files = sorted(
                list_texture_files(str(texture_dir)),
                key=lambda p: Path(p).name,
            )
        self.texture_rank_by_source_demo = (
            None
            if texture_rank_by_source_demo is None
            else {int(k): int(v) for k, v in texture_rank_by_source_demo.items()}
        )

    def _init_oracle_context(self):
        if self.oracle_camera is None:
            return OracleContext()
        return build_oracle_context(
            env=self.env,
            env_meta=self.env_meta,
            camera_names=self.camera_names,
            camera_name=self.oracle_camera,
            resolution=self.res,
            patch_size=self.oracle_patch_size,
        )

    def _table_texture_for_rank(self, demo_rank: int) -> Optional[str]:
        if self.table_texture_files is None:
            return None
        texture_idx = (int(demo_rank) // int(self.table_texture_every)) % len(
            self.table_texture_files
        )
        return self.table_texture_files[texture_idx]

    def rerender_demonstration(
        self,
        h5_file: h5py.File,
        ep: str,
        reset_every_step: bool = True,
        texture_demo_rank: Optional[int] = None,
    ):
        """Rerender one episode and collect frame-aligned oracle labels."""
        is_robosuite_env = EnvUtils.is_robosuite_env(self.env_meta)

        ep_grp = h5_file[f"data/{ep}"]
        states = ep_grp["states"][()]
        if "actions" not in ep_grp:
            raise KeyError(
                f"Episode {ep} missing raw action dataset 'actions'. "
                "Rerendering replays data/<demo>/actions."
            )
        actions = ep_grp["actions"][()]

        initial_state = {"states": states[0]}
        if is_robosuite_env:
            init_state = h5_file[f"data/{ep}"].attrs["model_file"]
            if self.table_texture_every is not None:
                demo_rank = (
                    self.rerender_counter
                    if texture_demo_rank is None
                    else int(texture_demo_rank)
                )
                texture = self._table_texture_for_rank(demo_rank)
                if texture is None:
                    raise RuntimeError("table texture scheduler was not initialized")
                init_state = apply_table_texture(
                    init_state,
                    texture_file=texture,
                )
            initial_state["model"] = init_state

        # Keep at most 7 dims before rollout (legacy behavior).
        if actions.ndim == 2 and actions.shape[1] > 7:
            actions = actions[:, :7]

        converted_actions, _ = convert_actions(
            self.action_conversion_env,
            states,
            actions,
        )
        actions = converted_actions["absolute"]

        collector = None
        if self.oracle_camera is not None:
            collector = OracleFrameCollector.build(
                oracle=self.oracle,
                horizon=int(actions.shape[0]),
                camera_name=self.oracle_camera,
                resolution=self.res,
                patch_size=self.oracle_patch_size,
                min_patch_area_fraction=self.oracle_min_patch_area_fraction,
                min_mask_pixels=self.oracle_min_mask_pixels,
            )

        # Rerender state-aligned frames and execute absolute actions one step.
        traj, success = extract_trajectory(
            env=self.env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            reset_every_step=reset_every_step,
            pre_step_callback=collector,
            verbose=False,
        )
        oracle_info = {} if collector is None else collector.as_arrays()
        return traj, success, oracle_info

    def render_to_cache(
        self,
        n_demo: Optional[int],
        start_index: int,
        jpeg_quality: int,
        lmdb_map_size_gb: int,
        commit_every: int,
        delta_horizons: List[int],
        demo_indices: Optional[List[int]] = None,
        show_progress: bool = True,
        progress_queue=None,
    ) -> None:
        """Process selected demos and write full cache outputs."""
        setup_task_embedding_cache()

        env_name = self.env_meta.get("env_name", None)
        if env_name is None:
            raise ValueError("env_name missing in env_meta")

        # Task metadata derived from env_name.
        task_meta = env_name_to_meta(env_name)
        task_instruction = str(task_meta["instruction"])
        robot_name = str(task_meta["robot"])
        robot_id_scalar = int(task_meta["robot_id"])
        task_emb = task_meta["task_embedding"].cpu().numpy().astype(np.float32)
        task_language_tokens = (
            task_meta["task_language_tokens"].cpu().numpy().astype(np.float32)
        )

        # LMDB init
        lmdb_path = os.path.join(self.out_dir, "images.lmdb")
        if os.path.exists(lmdb_path):
            os.remove(lmdb_path)

        env_lmdb = lmdb.open(
            lmdb_path,
            map_size=int(lmdb_map_size_gb * (1024**3)),
            subdir=False,
            readonly=False,
            meminit=False,
            map_async=True,
            max_dbs=1,
        )
        txn = env_lmdb.begin(write=True)
        put_count = 0

        lowdim_chunks: Dict[str, List[np.ndarray]] = defaultdict(list)
        abs_chunks: List[np.ndarray] = []
        episode_lengths: List[int] = []
        oracle_chunks: Dict[str, List[np.ndarray]] = defaultdict(list)

        # Episode-level outputs
        task_instructions: List[str] = []
        robot_names: List[str] = []
        robot_ids: List[int] = []
        task_embeddings: List[np.ndarray] = []
        task_language_token_embeddings: List[np.ndarray] = []
        source_demo_indices: List[int] = []
        table_texture_episode_files: List[str] = []

        global_step = 0
        kept_demos = 0

        with h5py.File(self.input_path, "r") as h5_file:
            all_demos = _sorted_demo_keys(h5_file)
            if demo_indices is None:
                start_index = int(start_index)
                if start_index < 0 or start_index >= len(all_demos):
                    raise ValueError(
                        f"start episode {start_index} out of range "
                        f"(episodes={len(all_demos)})"
                    )
                demos = all_demos[start_index:]
            else:
                demos = [f"demo_{int(i)}" for i in demo_indices]
                missing = [ep for ep in demos if ep not in h5_file["data"]]
                if missing:
                    raise KeyError(f"Requested demos not found in HDF5: {missing[:10]}")

            target_n_demo = None if n_demo is None else int(n_demo)
            if n_demo is None:
                n_demo = len(demos)

            pbar = (
                tqdm(total=n_demo, desc="Rerender -> LMDB cache", unit="ok")
                if show_progress
                else None
            )
            scanned = 0

            for ep in demos:
                scanned += 1
                if kept_demos >= int(n_demo):
                    break

                source_demo_idx = int(ep.split("_")[-1])
                texture_demo_rank = None
                texture_file = None
                if self.texture_rank_by_source_demo is not None:
                    texture_demo_rank = self.texture_rank_by_source_demo[source_demo_idx]
                if self.table_texture_every is not None:
                    demo_rank = (
                        kept_demos if texture_demo_rank is None else texture_demo_rank
                    )
                    texture_file = self._table_texture_for_rank(demo_rank)

                traj, success, oracle_info = self.rerender_demonstration(
                    h5_file,
                    ep,
                    texture_demo_rank=texture_demo_rank,
                )
                if not success:
                    if pbar is not None:
                        pbar.set_postfix_str(
                            f"kept={kept_demos}/{n_demo or '-'} scanned={scanned} skip_failed"
                        )
                    continue

                T = int(traj["actions"].shape[0])
                episode_lengths.append(T)
                for key, arr in oracle_info.items():
                    oracle_chunks[key].append(arr)

                # Episode-level arrays (one per written demo).
                task_instructions.append(task_instruction)
                robot_names.append(robot_name)
                robot_ids.append(robot_id_scalar)
                task_embeddings.append(task_emb)
                task_language_token_embeddings.append(task_language_tokens)
                source_demo_indices.append(source_demo_idx)
                if texture_file is not None:
                    table_texture_episode_files.append(Path(texture_file).name)

                # Actions: traj['actions'] are already absolute controller targets.
                abs_act = np.asarray(traj["actions"], dtype=np.float32)
                if abs_act.shape[0] != T:
                    raise ValueError("absolute action length mismatch")
                # Continuous gripper signal from qpos.
                if "robot0_gripper_qpos" not in traj["obs"]:
                    raise KeyError(
                        "traj['obs'] missing robot0_gripper_qpos (needed for gripper action)"
                    )
                gripper_qpos = np.asarray(
                    traj["obs"]["robot0_gripper_qpos"], dtype=np.float32
                )
                step = 2
                opening = np.abs(gripper_qpos[:, 0] - gripper_qpos[:, 1])  # (T,)
                gripper_act = np.concatenate(
                    [opening[step:], np.repeat(opening[-1], step)], axis=0
                )[:T]

                abs_act = np.concatenate([abs_act, gripper_act[:, None]], axis=-1)

                abs_posmat = action_to_posmat(abs_act)

                abs_chunks.append(abs_posmat)

                # Low-dimensional observations.
                obs = traj["obs"]

                for k, v in obs.items():
                    if "image" in k:
                        continue
                    arr = np.asarray(v, dtype=np.float32)
                    if arr.shape[0] != T:
                        raise ValueError(
                            f"lowdim '{k}' length mismatch: {arr.shape[0]} vs {T}"
                        )
                    lowdim_chunks[k].append(arr)

                # LMDB images.
                for t in range(T):
                    for cam_key in self.rgb_keys:
                        if cam_key not in obs:
                            available_keys = list(obs.keys())[:10]
                            raise KeyError(
                                f"Missing obs image key {cam_key!r}. "
                                f"Have keys: {available_keys} ..."
                            )
                        img = np.asarray(obs[cam_key][t], dtype=np.uint8)  # RGB HWC

                        if img.shape[0] != self.res or img.shape[1] != self.res:
                            img = cv2.resize(
                                img, (self.res, self.res), interpolation=cv2.INTER_AREA
                            )

                        jpg = encode_rgb_to_jpg_bytes(img, quality=jpeg_quality)
                        key = f"{cam_key}/{global_step:08d}".encode("ascii")
                        txn.put(key, jpg)
                        put_count += 1

                        if put_count % int(commit_every) == 0:
                            txn.commit()
                            txn = env_lmdb.begin(write=True)

                    global_step += 1

                kept_demos += 1
                self.rerender_counter = kept_demos
                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "event": PROGRESS_EVENT_DEMO_DONE,
                            "n_samples": T,
                        }
                    )
                if pbar is not None:
                    pbar.update(1)  # only on kept/success
                failed = scanned - kept_demos
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"kept={kept_demos}/{n_demo or '-'} scanned={scanned} failed={failed}"
                    )
        if pbar is not None:
            pbar.close()
        if target_n_demo is not None and kept_demos < target_n_demo:
            txn.abort()
            env_lmdb.close()
            raise RuntimeError(
                f"Requested {target_n_demo} successful demos, but only "
                f"{kept_demos} succeeded before source demos were exhausted."
            )
        # Finalize LMDB.
        txn.put(b"__len__", str(global_step).encode("ascii"))
        txn.commit()
        env_lmdb.sync()
        env_lmdb.close()

        if kept_demos == 0:
            raise RuntimeError("No demos were written (all failed or none selected).")

        # Pack non-image arrays into one archive.
        arrays = {}
        lowdim_keys = sorted(list(lowdim_chunks.keys()))
        for k in lowdim_keys:
            arrays[archive_key(os.path.join("lowdim", f"{k}.npy"))] = np.concatenate(
                lowdim_chunks[k], axis=0
            ).astype(np.float32)

        abs_all = np.concatenate(abs_chunks, axis=0).astype(np.float32)
        arrays[archive_key(os.path.join("action", "absolute_action.npy"))] = abs_all
        for delta_horizon in sorted({int(h) for h in delta_horizons}):
            if delta_horizon < 1:
                raise ValueError(f"delta horizon must be >= 1, got {delta_horizon}")
            arrays[
                archive_key(os.path.join("action", f"delta_action_h{delta_horizon}.npy"))
            ] = absolute_posmat_to_delta_chunks(
                eef_pos=arrays[
                    archive_key(os.path.join("lowdim", "robot0_eef_pos.npy"))
                ],
                eef_rot=arrays[
                    archive_key(os.path.join("lowdim", "robot0_eef_rot.npy"))
                ],
                action_posmat=abs_all,
                horizon=delta_horizon,
            )
        oracle_arrays = build_oracle_arrays(
            oracle_chunks=oracle_chunks,
            expected_steps=int(sum(episode_lengths)),
        )
        for key, arr in oracle_arrays.items():
            arrays[archive_key(os.path.join("oracle", f"{key}.npy"))] = arr
        oracle_keys = list(oracle_arrays.keys())
        arrays[archive_key(os.path.join("lowdim", "task_embedding.npy"))] = np.stack(
            task_embeddings, axis=0
        ).astype(np.float32)
        arrays[archive_key(os.path.join("lowdim", "task_language_tokens.npy"))] = (
            np.stack(task_language_token_embeddings, axis=0).astype(np.float32)
        )
        arrays[archive_key(os.path.join("lowdim", "task_id.npy"))] = np.zeros(
            (len(episode_lengths),), dtype=np.int64
        )
        arrays[archive_key(os.path.join("lowdim", "robot_id.npy"))] = np.asarray(
            robot_ids, dtype=np.int64
        )
        arrays[archive_key("task_instructions.npy")] = np.asarray(
            task_instructions,
            dtype=object,
        )
        np.savez(archive_path(self.out_dir), **arrays)

        meta = {
            "cache_format": "lmdb_npz_v1",
            "env_meta": _json_safe(self.source_env_meta),
            "env_name": str(self.env_meta.get("env_name", "")),
            "rgb_keys": list(self.rgb_keys),
            "lowdim_keys": lowdim_keys,
            "episode_lengths": list(map(int, episode_lengths)),
            "n_demo": int(len(episode_lengths)),
            "n_samples": int(sum(episode_lengths)),
            "image_size": int(self.res),
            "robot_names": robot_names,
            "source_demo_indices": list(map(int, source_demo_indices)),
            "delta_action_horizons": sorted({int(h) for h in delta_horizons}),
        }
        if self.table_texture_every is not None:
            meta["table_texture"] = {
                "mode": "ranked_table_texture",
                "demos_per_texture": int(self.table_texture_every),
                "texture_dir": str((Path(TEXTURES_DIR) / "train").resolve()),
                "files": [Path(path).name for path in self.table_texture_files],
                "episode_files": table_texture_episode_files,
            }
        if oracle_keys:
            meta["oracle_keys"] = oracle_keys
            meta["oracle_camera"] = self.oracle_camera
            meta["oracle_patch_size"] = int(self.oracle_patch_size)
            meta["oracle_min_patch_area_fraction"] = float(
                self.oracle_min_patch_area_fraction
            )
            meta["oracle_min_mask_pixels"] = int(self.oracle_min_mask_pixels)
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        with open(os.path.join(self.out_dir, "build_done.flag"), "w") as f:
            f.write("build completed\n")

        print(
            f"[built] out_dir={self.out_dir} demos={meta['n_demo']} samples={meta['n_samples']}"
        )


def _rerender_worker(
    *,
    worker_id: int,
    dataset_path: str,
    out_cache_dir: str,
    camera_resolution: int,
    table_texture_every: Optional[int],
    texture_rank_by_source_demo: Optional[Dict[int, int]],
    oracle_camera: Optional[str],
    oracle_patch_size: int,
    oracle_min_patch_area_fraction: float,
    oracle_min_mask_pixels: int,
    demo_indices: List[int],
    delta_horizons: List[int],
    overwrite: bool,
    progress_queue=None,
) -> dict:
    """Worker entrypoint for shard-based parallel rerender."""
    builder = DatasetRerenderToCache(
        dataset_path=dataset_path,
        out_cache_dir=out_cache_dir,
        camera_resolution=camera_resolution,
        table_texture_every=table_texture_every,
        texture_rank_by_source_demo=texture_rank_by_source_demo,
        oracle_camera=oracle_camera,
        oracle_patch_size=oracle_patch_size,
        oracle_min_patch_area_fraction=oracle_min_patch_area_fraction,
        oracle_min_mask_pixels=oracle_min_mask_pixels,
        overwrite=overwrite,
    )
    try:
        builder.render_to_cache(
            n_demo=None,
            start_index=0,
            jpeg_quality=JPEG_QUALITY_DEFAULT,
            lmdb_map_size_gb=LMDB_MAP_SIZE_GB_DEFAULT,
            commit_every=COMMIT_EVERY_DEFAULT,
            delta_horizons=delta_horizons,
            demo_indices=demo_indices,
            show_progress=False,
            progress_queue=progress_queue,
        )
    except RuntimeError as exc:
        if "No demos were written" not in str(exc):
            raise
        return {
            "worker_id": int(worker_id),
            "out_dir": str(out_cache_dir),
            "n_demo": 0,
            "n_samples": 0,
        }
    with open(os.path.join(out_cache_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    return {
        "worker_id": int(worker_id),
        "out_dir": str(out_cache_dir),
        "n_demo": int(meta["n_demo"]),
        "n_samples": int(meta["n_samples"]),
    }


def _run_parallel_rerender(
    *,
    dataset_path: str,
    output_dir: str,
    camera_resolution: int,
    table_texture_every: Optional[int],
    oracle_camera: Optional[str],
    oracle_patch_size: int,
    oracle_min_patch_area_fraction: float,
    oracle_min_mask_pixels: int,
    start_index: int,
    n_demo: Optional[int],
    num_workers: int,
    delta_horizons: List[int],
    overwrite: bool,
) -> None:
    """Rerender source demos in parallel shard caches and merge them."""
    from .merge_lmdb_caches import merge_caches

    output_path = Path(output_dir).expanduser().resolve()
    _remove_existing_output_dir(output_path, overwrite=overwrite)

    shard_root = output_path.parent / f".{output_path.name}_worker_shards"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    selected = _source_demo_indices(dataset_path, start_index, None)
    texture_rank_by_source_demo = None
    if table_texture_every is not None:
        texture_rank_by_source_demo = {
            int(source_idx): rank for rank, source_idx in enumerate(selected)
        }
    target_successes = len(selected) if n_demo is None else int(n_demo)
    if target_successes < 1:
        raise ValueError(f"n_demo must be >= 1, got {n_demo}")
    if n_demo is not None and target_successes > len(selected):
        raise ValueError(
            f"n_demo={target_successes} exceeds remaining source demos={len(selected)}"
        )
    print(
        f"[parallel] rerendering {target_successes} successful demos "
        f"with up to {int(num_workers)} workers",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    progress_queue = manager.Queue()
    planned_shards = []
    built_shard_paths = set()
    try:
        cursor = 0
        round_idx = 0
        success_count = 0
        samples = 0
        with tqdm(total=target_successes, desc="parallel rerender", unit="ok") as pbar:
            while cursor < len(selected) and success_count < target_successes:
                remaining = target_successes - success_count
                candidates = selected[cursor : cursor + remaining]
                cursor += len(candidates)
                chunks = _split_contiguous(candidates, num_workers)
                round_root = shard_root / f"round_{round_idx:03d}"
                shard_dirs = [
                    round_root / f"worker_{i:03d}" for i in range(len(chunks))
                ]
                planned_shards.extend(shard_dirs)
                futures = []
                with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as pool:
                    for worker_id, (indices, shard_dir) in enumerate(
                        zip(chunks, shard_dirs)
                    ):
                        futures.append(
                            pool.submit(
                                _rerender_worker,
                                worker_id=worker_id,
                                dataset_path=dataset_path,
                                out_cache_dir=str(shard_dir),
                                camera_resolution=camera_resolution,
                                table_texture_every=table_texture_every,
                                texture_rank_by_source_demo=texture_rank_by_source_demo,
                                oracle_camera=oracle_camera,
                                oracle_patch_size=oracle_patch_size,
                                oracle_min_patch_area_fraction=(
                                    oracle_min_patch_area_fraction
                                ),
                                oracle_min_mask_pixels=oracle_min_mask_pixels,
                                demo_indices=indices,
                                delta_horizons=delta_horizons,
                                overwrite=True,
                                progress_queue=progress_queue,
                            )
                        )

                    done = set()
                    while len(done) < len(futures):
                        for future in futures:
                            if future in done or not future.done():
                                continue
                            result = future.result()
                            done.add(future)
                            if int(result["n_demo"]) > 0:
                                built_shard_paths.add(Path(result["out_dir"]).resolve())
                            print(
                                f"[parallel] worker {result['worker_id']:03d} built "
                                f"{result['n_demo']} demos / {result['n_samples']} samples",
                                flush=True,
                            )

                        samples, success_count, drained = _drain_progress_events(
                            progress_queue,
                            pbar,
                            samples,
                            success_count,
                        )
                        if not drained and len(done) < len(futures):
                            time.sleep(PROGRESS_POLL_INTERVAL_SEC)

                    samples, success_count, _ = _drain_progress_events(
                        progress_queue,
                        pbar,
                        samples,
                        success_count,
                    )
                round_idx += 1
    finally:
        manager.shutdown()

    built_shards = [
        shard_dir
        for shard_dir in planned_shards
        if shard_dir.resolve() in built_shard_paths
    ]
    if not built_shards:
        raise RuntimeError("Parallel rerender produced no shard caches")
    if n_demo is not None and success_count < target_successes:
        raise RuntimeError(
            f"[parallel] requested {target_successes} successful demos, "
            f"built {success_count}"
        )

    episode_indices_per_input = None
    if n_demo is not None:
        successful = []
        for input_idx, shard_dir in enumerate(built_shards):
            with (shard_dir / "meta.json").open("r") as f:
                meta = json.load(f)
            for episode_idx, source_idx in enumerate(meta["source_demo_indices"]):
                successful.append((int(source_idx), input_idx, int(episode_idx)))
        successful.sort()
        episode_indices_per_input = [[] for _ in built_shards]
        for _, input_idx, episode_idx in successful[:target_successes]:
            episode_indices_per_input[input_idx].append(episode_idx)

    merge_caches(
        in_dirs=built_shards,
        out_dir=output_path,
        n_demo_per_input=None,
        source_task_names=[Path(dataset_path).parent.name] * len(built_shards),
        episode_indices_per_input=episode_indices_per_input,
        overwrite=True,
        lmdb_map_size_gb=LMDB_MAP_SIZE_GB_DEFAULT,
        commit_every=COMMIT_EVERY_DEFAULT,
        delta_horizons=delta_horizons,
    )

    shutil.rmtree(shard_root)


def _run_single_rerender(
    *,
    dataset_path: str,
    output_dir: str,
    camera_resolution: int,
    table_texture_every: Optional[int],
    oracle_camera: Optional[str],
    oracle_patch_size: int,
    oracle_min_patch_area_fraction: float,
    oracle_min_mask_pixels: int,
    start_index: int,
    n_demo: Optional[int],
    num_workers: int,
    delta_horizons: List[int],
    overwrite: bool,
) -> None:
    """Rerender one raw HDF5 dataset to one LMDB cache."""
    if int(num_workers) > 1:
        _run_parallel_rerender(
            dataset_path=dataset_path,
            output_dir=output_dir,
            camera_resolution=camera_resolution,
            table_texture_every=table_texture_every,
            oracle_camera=oracle_camera,
            oracle_patch_size=oracle_patch_size,
            oracle_min_patch_area_fraction=oracle_min_patch_area_fraction,
            oracle_min_mask_pixels=oracle_min_mask_pixels,
            start_index=start_index,
            n_demo=n_demo,
            num_workers=num_workers,
            delta_horizons=delta_horizons,
            overwrite=overwrite,
        )
        return

    builder = DatasetRerenderToCache(
        dataset_path=dataset_path,
        out_cache_dir=output_dir,
        camera_resolution=camera_resolution,
        table_texture_every=table_texture_every,
        oracle_camera=oracle_camera,
        oracle_patch_size=oracle_patch_size,
        oracle_min_patch_area_fraction=oracle_min_patch_area_fraction,
        oracle_min_mask_pixels=oracle_min_mask_pixels,
        overwrite=overwrite,
    )

    builder.render_to_cache(
        n_demo=n_demo,
        start_index=start_index,
        jpeg_quality=JPEG_QUALITY_DEFAULT,
        lmdb_map_size_gb=LMDB_MAP_SIZE_GB_DEFAULT,
        commit_every=COMMIT_EVERY_DEFAULT,
        delta_horizons=delta_horizons,
    )


def _rerender_kwargs_from_args(args: argparse.Namespace) -> dict:
    if args.table_texture_every is not None and int(args.table_texture_every) < 1:
        raise ValueError(
            f"--table-texture-every must be >= 1, got {args.table_texture_every}"
        )
    oracle_camera = str(args.oracle_camera).strip()
    if oracle_camera == "":
        oracle_camera = None
    return {
        "camera_resolution": 256,
        "table_texture_every": args.table_texture_every,
        "oracle_camera": oracle_camera,
        "oracle_patch_size": args.oracle_patch_size,
        "oracle_min_patch_area_fraction": args.oracle_min_patch_area_fraction,
        "oracle_min_mask_pixels": args.oracle_min_mask_pixels,
        "start_index": 0,
        "n_demo": args.n_demo,
        "num_workers": args.num_workers,
        "delta_horizons": [16],
        "overwrite": args.overwrite,
    }


def _validate_bulk_args(args: argparse.Namespace) -> None:
    if args.datasets_root is None:
        raise ValueError(
            "Provide either --dataset for one HDF5 or --datasets-root for bulk mode."
        )


def _run_bulk_rerender_from_args(args: argparse.Namespace) -> None:
    _validate_bulk_args(args)
    if args.output_dir is not None:
        raise ValueError("--output-dir is only valid with --dataset")

    datasets = _discover_task_datasets(args.datasets_root, tasks=args.tasks)
    rerender_kwargs = _rerender_kwargs_from_args(args)

    print(f"[bulk] rerendering {len(datasets)} task datasets", flush=True)
    for task_idx, (task_name, dataset_path) in enumerate(datasets, start=1):
        base_cache_dir = default_cache_dir_for_dataset(
            dataset_path,
            output_root=args.datasets_root,
        )
        output_dir = _sibling_cache_dir_with_suffix(base_cache_dir, args.output_suffix)
        print(
            f"[bulk] ({task_idx}/{len(datasets)}) {task_name}: "
            f"{dataset_path} -> {output_dir}",
            flush=True,
        )
        _run_single_rerender(
            dataset_path=dataset_path,
            output_dir=output_dir,
            **rerender_kwargs,
        )


def _run_dataset_rerender_from_args(args: argparse.Namespace) -> None:
    dataset_path = str(Path(args.dataset).expanduser().resolve())
    base_cache_dir = default_cache_dir_for_dataset(dataset_path)
    output_dir = (
        str(Path(args.output_dir).expanduser().resolve())
        if args.output_dir is not None
        else _sibling_cache_dir_with_suffix(base_cache_dir, args.output_suffix)
    )
    _run_single_rerender(
        dataset_path=dataset_path,
        output_dir=output_dir,
        **_rerender_kwargs_from_args(args),
    )


def main():
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(
        description="Rerender raw HDF5 demonstrations into self-contained LMDB caches."
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to one input .hdf5 dataset.",
    )
    ap.add_argument(
        "--datasets-root",
        type=str,
        default=None,
        help=(
            "Root containing task folders. In bulk mode, tasks are expected at "
            "<datasets_root>/<task>/<task>.hdf5."
        ),
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names under --datasets-root. Defaults to all task folders.",
    )
    ap.add_argument(
        "--n-demo",
        type=int,
        default=None,
        help="Number of successful demos to write.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output cache directory for --dataset mode.",
    )
    ap.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to the default cache directory name.",
    )
    ap.add_argument(
        "--table-texture-every",
        type=int,
        default=None,
        help="Apply a new table texture every N successful demos.",
    )
    ap.add_argument(
        "--oracle-camera",
        type=str,
        default="agentview",
        help=(
            "Camera used to cache oracle focus labels. Set to '' to disable oracle "
            "label collection."
        ),
    )
    ap.add_argument(
        "--oracle-patch-size",
        type=int,
        default=16,
        help="Patch size for cached oracle target masks.",
    )
    ap.add_argument(
        "--oracle-min-patch-area-fraction",
        type=float,
        default=0.05,
        help="Minimum visible area fraction for a target patch to be marked.",
    )
    ap.add_argument(
        "--oracle-min-mask-pixels",
        type=int,
        default=16,
        help="Drop oracle segmentation masks with fewer visible target pixels than this.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel rerender workers.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output cache directories without prompting.",
    )

    args = ap.parse_args()

    if args.dataset is None:
        _run_bulk_rerender_from_args(args)
        return

    _run_dataset_rerender_from_args(args)


if __name__ == "__main__":
    main()
