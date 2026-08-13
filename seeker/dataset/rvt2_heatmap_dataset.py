"""Dataset and heuristic label construction for RVT2Heatmap prediction."""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from seeker.dataset.mimicgen_dataset import MimicGenDataset
from seeker.focus_toolbox.rvt2_heatmap import (
    _project_world_xyz_to_row_col,
    discover_rvt2_heatmap_keypoints,
    gripper_open_from_signal,
)
from seeker import MIMICGEN_DATASETS_DIR, REPO_ROOT, WEIGHTS_DIR


JOINT_VELOCITY_KEY = "robot0_joint_vel"

RVT2_HEATMAP_SHAPE_META = {
    "obs": {
        "agentview_image": {"shape": [3, 224, 224], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
        "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
    },
    "action": {"shape": [10]},
}


@dataclass(frozen=True)
class PatchRecord:
    episode_index: int
    active_frame_index: int
    source_frame: int
    target_frame: int
    target_patch: int
    gripper_context: np.ndarray


class RVT2HeatmapPatchDataset(Dataset):
    """Lazily load cached images and attach RVT2Heatmap heuristic patch labels."""

    def __init__(
        self,
        dataset: MimicGenDataset,
        records: list[PatchRecord],
        gripper_mean: np.ndarray,
        gripper_std: np.ndarray,
    ) -> None:
        self.dataset = dataset
        self.records = records
        self.gripper_mean = np.asarray(gripper_mean, dtype=np.float32)
        self.gripper_std = np.asarray(gripper_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def task_sample_ranges(self) -> dict[int, list[tuple[int, int]]]:
        task_to_ranges = defaultdict(list)
        current_episode = None
        start = 0
        for idx, record in enumerate(self.records):
            if current_episode is None:
                current_episode = int(record.episode_index)
            elif int(record.episode_index) != current_episode:
                task_id = int(self.dataset.task_id_episode[current_episode])
                task_to_ranges[task_id].append((start, idx))
                current_episode = int(record.episode_index)
                start = idx
        if current_episode is not None:
            task_id = int(self.dataset.task_id_episode[current_episode])
            task_to_ranges[task_id].append((start, len(self.records)))
        return dict(task_to_ranges)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[int(idx)]
        obs = self.dataset.get_obs(
            record.episode_index,
            [record.active_frame_index],
        )
        image = torch.from_numpy(obs["agentview_image"][0])
        gripper = (record.gripper_context - self.gripper_mean) / self.gripper_std
        item = {
            "image": image,
            "gripper": torch.from_numpy(gripper.astype(np.float32)),
            "task_embedding": torch.from_numpy(
                obs["task_embedding"][0].astype(np.float32)
            ),
            "robot_id": torch.tensor(
                int(np.asarray(obs["robot_id"][0]).reshape(-1)[0]),
                dtype=torch.long,
            ),
            "eef_pos": torch.from_numpy(obs["robot0_eef_pos"][0].astype(np.float32)),
            "target_patch": torch.tensor(record.target_patch, dtype=torch.long),
            "episode_index": torch.tensor(record.episode_index, dtype=torch.long),
            "source_frame": torch.tensor(record.source_frame, dtype=torch.long),
            "target_frame": torch.tensor(record.target_frame, dtype=torch.long),
        }
        if "task_language_tokens" not in obs:
            raise ValueError(
                "RVT2Heatmap training requires lowdim/task_language_tokens.npy. "
                "Rerender or remerge the cache with the current task metadata code."
            )
        item["task_language_tokens"] = torch.from_numpy(
            obs["task_language_tokens"][0].astype(np.float32)
        )
        return item


def resolve_rvt2_heatmap_dataset_path(
    task_name: str | None,
    dataset_path: str | None,
) -> Path:
    if dataset_path:
        path = Path(dataset_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    if not task_name:
        raise ValueError("Provide either --task-name or --dataset-path.")
    return (MIMICGEN_DATASETS_DIR / task_name / f"{task_name}_lmdb").resolve()


def resolve_dino_checkpoint(path: str | None) -> Path:
    if path:
        ckpt_path = Path(path).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / ckpt_path
        return ckpt_path.resolve()
    return (WEIGHTS_DIR / "dinov3_vits16plus.pth").resolve()


def _row_col_to_patch(
    row_col: np.ndarray,
    *,
    source_image_size: int,
    dino_image_size: int,
    patch_size: int,
) -> int:
    grid = dino_image_size // patch_size
    scale = float(dino_image_size) / float(source_image_size)
    row = int(np.clip(math.floor(float(row_col[0]) * scale / patch_size), 0, grid - 1))
    col = int(np.clip(math.floor(float(row_col[1]) * scale / patch_size), 0, grid - 1))
    return row * grid + col


def _gripper_opening(gripper_qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(gripper_qpos, dtype=np.float32)
    if qpos.ndim == 2 and qpos.shape[1] >= 2:
        return (np.abs(qpos[:, 0] - qpos[:, 1]) - 1.0).reshape(-1, 1)
    return qpos.reshape(qpos.shape[0], -1)[:, :1]


def build_rvt2_heatmap_patch_records(
    dataset: MimicGenDataset,
    *,
    episode_indices: list[int],
    camera: str,
    dino_image_size: int,
    patch_size: int,
    joint_vel_atol: float,
    stopped_buffer_len: int,
    include_final: bool,
    mute_initial_gripper_open: bool,
    show_progress: bool,
) -> tuple[list[PatchRecord], dict]:
    if dataset.oracle_info is None:
        raise ValueError(
            "Cache is missing oracle info. Rerender the cache with the current "
            "rerender_dataset.py so camera matrices are exported."
        )
    camera_key = f"camera_matrix_{camera}"
    if camera_key not in dataset.oracle_info:
        available = ", ".join(sorted(dataset.oracle_info.keys()))
        raise ValueError(
            f"Cache is missing oracle/{camera_key}.npy. Available oracle keys: {available}"
        )

    source_image_size = int(dataset.meta.get("image_size", dino_image_size))
    if source_image_size <= 0:
        source_image_size = dino_image_size

    records: list[PatchRecord] = []
    stats = {
        "episodes": 0,
        "keypoints": 0,
        "frames_with_next_keypoint": 0,
        "records": 0,
        "projection_failed": 0,
        "mute_initial_gripper_open": bool(mute_initial_gripper_open),
        "velocity_sources": {},
    }

    camera_matrices = np.asarray(dataset.oracle_info[camera_key], dtype=np.float32)

    episode_iter = tqdm(
        episode_indices,
        desc="label episodes",
        file=sys.stdout,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for episode_index in episode_iter:
        active_indices = dataset.episode_active_indices(episode_index)
        global_indices = dataset.active_to_global_indices(active_indices)
        eef_pos = dataset.lowdim["robot0_eef_pos"][global_indices].astype(np.float32)
        gripper_qpos = dataset.lowdim["robot0_gripper_qpos"][global_indices].astype(
            np.float32
        )

        joint_velocities, velocity_source = dataset.optional_episode_lowdim(
            episode_index, JOINT_VELOCITY_KEY
        )
        if joint_velocities is None:
            raise ValueError(
                "RVT2Heatmap heuristic keypoint labels require cached joint velocities; "
                f"episode {episode_index} is missing lowdim/{JOINT_VELOCITY_KEY}.npy."
            )
        stats["velocity_sources"][velocity_source] = (
            stats["velocity_sources"].get(velocity_source, 0) + 1
        )

        keypoints, _ = discover_rvt2_heatmap_keypoints(
            joint_velocities,
            gripper_open_from_signal(gripper_qpos),
            atol=joint_vel_atol,
            stopped_buffer_len=stopped_buffer_len,
            include_gripper_changes=True,
            include_final=include_final,
            mute_initial_gripper_open=mute_initial_gripper_open,
        )
        keypoints = np.asarray(keypoints, dtype=np.int64)
        stats["episodes"] += 1
        stats["keypoints"] += int(keypoints.shape[0])
        if keypoints.size == 0:
            continue

        gripper_context = _gripper_opening(gripper_qpos).astype(np.float32)

        for t in range(int(eef_pos.shape[0])):
            next_pos = int(np.searchsorted(keypoints, t, side="right"))
            if next_pos >= keypoints.shape[0]:
                continue
            target_t = int(keypoints[next_pos])
            stats["frames_with_next_keypoint"] += 1
            row_col = _project_world_xyz_to_row_col(
                eef_pos[target_t],
                camera_matrices[int(global_indices[t])],
                source_image_size,
            )
            if row_col is None:
                stats["projection_failed"] += 1
                continue
            target_patch = _row_col_to_patch(
                row_col,
                source_image_size=source_image_size,
                dino_image_size=dino_image_size,
                patch_size=patch_size,
            )
            records.append(
                PatchRecord(
                    episode_index=episode_index,
                    active_frame_index=int(active_indices[t]),
                    source_frame=t,
                    target_frame=target_t,
                    target_patch=int(target_patch),
                    gripper_context=gripper_context[t].copy(),
                )
            )

    stats["records"] = len(records)
    return records, stats


def split_patch_records(
    records: list[PatchRecord],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[PatchRecord], list[PatchRecord]]:
    episodes = sorted({record.episode_index for record in records})
    rng = random.Random(seed)
    rng.shuffle(episodes)
    if len(episodes) <= 1 or val_ratio <= 0:
        return records, []
    n_val = max(1, int(round(len(episodes) * val_ratio)))
    n_val = min(n_val, len(episodes) - 1)
    val_episodes = set(episodes[:n_val])
    train_records = [r for r in records if r.episode_index not in val_episodes]
    val_records = [r for r in records if r.episode_index in val_episodes]
    return train_records, val_records


def gripper_stats(records: list[PatchRecord]) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("Cannot compute gripper stats from an empty record set.")
    arr = np.stack([record.gripper_context for record in records], axis=0).astype(
        np.float32
    )
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)
