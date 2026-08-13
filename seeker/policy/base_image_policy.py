"""Shared base interface for image-conditioned policies."""

from typing import Dict
import torch
from seeker.model.common.module_attr_mixin import ModuleAttrMixin
from seeker.model.common.normalizer import LinearNormalizer


class BaseImagePolicy(ModuleAttrMixin):
    """Abstract policy interface used by training and inference policies."""

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run policy inference.

        Args:
            obs_dict: Observation dictionary with shape `[B, To, ...]` tensors.
        Returns:
            Dict containing at least `action` with shape `[B, Ta, Da]`.
        """
        raise NotImplementedError()

    def reset(self):
        """Reset state for stateful policies."""
        pass

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set data normalizer used by the policy."""
        raise NotImplementedError()
