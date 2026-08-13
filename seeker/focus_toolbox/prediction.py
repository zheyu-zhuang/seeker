"""Shared visual-focus prediction containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class VisualFocusPrediction:
    """Where a focus model predicts the policy should look in an image."""

    box_px: torch.Tensor
    mask_grid: torch.Tensor | None = None
    heatmap: torch.Tensor | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, *args: Any, **kwargs: Any) -> "VisualFocusPrediction":
        """Return a copy with tensor fields moved like ``torch.Tensor.to``."""
        return VisualFocusPrediction(
            box_px=self.box_px.to(*args, **kwargs),
            mask_grid=(
                None if self.mask_grid is None else self.mask_grid.to(*args, **kwargs)
            ),
            heatmap=None if self.heatmap is None else self.heatmap.to(*args, **kwargs),
            source=self.source,
            metadata=dict(self.metadata),
        )


@dataclass
class VisualFocusRecord:
    """One focus prediction attached to a source, camera view, and timestep."""

    source: str
    view: str
    timestep: int | None
    prediction: VisualFocusPrediction
    image_size: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)
