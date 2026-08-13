"""Training checkpoint and JSON-lines logging helpers."""

from __future__ import annotations

import json
import numbers
import os
from typing import Dict, Optional


class TopKCheckpointManager:
    """Track top-k checkpoint paths by a scalar metric."""

    def __init__(
        self,
        save_dir,
        monitor_key: str,
        mode="min",
        k=1,
        format_str="epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt",
    ):
        assert mode in ["max", "min"]
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == "max":
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None

        del self.path_value_map[delete_path]
        self.path_value_map[ckpt_path] = value

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if os.path.exists(delete_path):
            os.remove(delete_path)
        return ckpt_path


class JsonLogger:
    """Append numeric training metrics as JSON lines."""

    def __init__(self, path: str):
        self.path = path
        self.file = None

    def __enter__(self):
        self.file = open(self.path, "a", buffering=1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        self.file = None

    def log(self, data: dict):
        filtered_data = {}
        for key, value in data.items():
            if isinstance(value, numbers.Integral):
                filtered_data[key] = int(value)
            elif isinstance(value, numbers.Number):
                filtered_data[key] = float(value)
        self.file.write(json.dumps(filtered_data).replace("\n", "") + "\n")
