"""Samplers for multi-task dataset training."""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
from torch.utils.data import Sampler


class TaskBalancedSampler(Sampler[int]):
    """Sample task -> episode -> timestep with a fixed per-epoch budget."""

    def __init__(
        self,
        task_to_ranges: dict[int, list[tuple[int, int]]],
        samples_per_epoch: int,
        *,
        seed: int = 0,
    ) -> None:
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

        if self.samples_per_epoch < 1:
            raise ValueError(f"samples_per_epoch must be >= 1, got {samples_per_epoch}")

        self.num_samples = self.samples_per_epoch
        self.task_to_ranges = task_to_ranges
        self.task_ids = np.asarray(sorted(self.task_to_ranges.keys()), dtype=np.int64)
        if self.task_ids.size == 0:
            raise ValueError("No task-balanced sample ranges available")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.num_samples)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch * 1009)
        repeats = self.num_samples // int(self.task_ids.size)
        remainder = self.num_samples % int(self.task_ids.size)
        tasks = np.tile(self.task_ids, repeats)
        if remainder:
            tasks = np.concatenate(
                [
                    tasks,
                    rng.choice(self.task_ids, size=remainder, replace=False),
                ],
                axis=0,
            )
        rng.shuffle(tasks)

        for task_id in tasks.tolist():
            ranges = self.task_to_ranges[int(task_id)]
            ep_range = ranges[int(rng.integers(0, len(ranges)))]
            yield int(rng.integers(ep_range[0], ep_range[1]))


def build_task_balanced_sampler(
    dataset,
    cfg: Optional[dict],
    *,
    seed: int,
) -> Optional[TaskBalancedSampler]:
    """Create a task-balanced sampler from an optional dataloader sampler config."""
    if not cfg:
        return None
    sampler_type = str(cfg.get("type", "")).strip().lower()
    if sampler_type not in ("task_balanced", "task-balanced"):
        raise ValueError(f"Unsupported sampler type: {sampler_type!r}")
    if "samples_per_epoch" not in cfg:
        raise ValueError("Task-balanced sampler requires samples_per_epoch")

    return TaskBalancedSampler(
        dataset.task_sample_ranges(),
        samples_per_epoch=int(cfg["samples_per_epoch"]),
        seed=int(cfg.get("seed", seed)),
    )
