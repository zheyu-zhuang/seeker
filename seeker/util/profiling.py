"""Lightweight profiling helpers for inference-time processing measurements."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import DefaultDict, Dict, Iterator, Optional

import torch


def _sync_if_needed(device: Optional[torch.device]) -> None:
    """Synchronize CUDA work around timing boundaries when needed."""
    if device is None:
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class InferenceProfiler:
    """Collect per-section processing-time measurements in milliseconds."""

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._records: DefaultDict[str, list[float]] = defaultdict(list)

    def set_enabled(self, enabled: bool, reset: bool = False) -> None:
        self.enabled = bool(enabled)
        if reset:
            self.reset()

    def reset(self) -> None:
        self._records.clear()

    @contextmanager
    def section(
        self, name: str, device: Optional[torch.device] = None
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        _sync_if_needed(device)
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync_if_needed(device)
            self._records[name].append((time.perf_counter() - start) * 1000.0)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name in sorted(self._records.keys()):
            values = self._records[name]
            if not values:
                continue
            out[name] = {
                "count": len(values),
                "avg_ms": round(sum(values) / len(values), 3),
                "total_ms": round(sum(values), 3),
            }
        return out
