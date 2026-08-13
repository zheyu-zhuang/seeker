"""Shared observation input preprocessing and validation utilities.

This module defines the processor used by model encoders to transform raw
observation dictionaries into normalized, flattened `EncoderInputs` tensors
consumed by Seeker-based encoder variants.
"""

from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from seeker.util.formatting import find_all_keys
from seeker.util.image_ops import normalize_imagenet, resize_image
from seeker.util.task_meta import NUM_ROBOTS


@dataclass
class EncoderInputs:
    agentview: torch.Tensor  # [B*T, 3, vit_in, vit_in]
    eye_in_hand: Optional[torch.Tensor]  # [B*T, 3, vit_in, vit_in] or None
    proprio: torch.Tensor  # [B*T, 11]
    robot_id: torch.Tensor  # [B*T, NUM_ROBOTS]
    composer_in: dict  # dict of tensors [B*T, ...]
    task_embedding: torch.Tensor  # [B*T, D]
    T: int


class ObsInputProcessor(nn.Module):
    """Common observation input path for policy-facing encoders.

    Responsibilities:
    - discover and cache observation keys;
    - flatten temporal observations from [B, T, ...] to [B*T, ...];
    - normalize observations via the provided normalizer;
    - build validated `composer_in` tensors used by Seeker components.
    """

    def __init__(self, input_res: Optional[int] = None, enable_eih: Optional[bool] = None):
        super().__init__()

        self.input_res = None if input_res is None else int(input_res)
        self.enable_eih = None if enable_eih is None else bool(enable_eih)
        self.obs_keys = None
        self.num_robots = NUM_ROBOTS

    def set_normalizer(self, normalizer: nn.Module):
        raise NotImplementedError

    def _init_obs_keys(self, obs: dict):
        if self.obs_keys is not None:
            return

        self.obs_keys = {
            "agentview": find_all_keys("agentview", obs.keys(), only_one_match=True),
            "in_hand": (
                find_all_keys("in_hand", obs.keys(), only_one_match=True)
                if self.enable_eih
                else None
            ),
            "eef_pos": find_all_keys("eef_pos", obs.keys(), only_one_match=True),
            "eef_rot": find_all_keys("eef_rot", obs.keys(), only_one_match=True),
            "gripper": find_all_keys("gripper", obs.keys(), only_one_match=True),
        }

    def obs_to_input(
        self,
        obs: dict,
        normalizer,
        resize: bool = False,
        composer_normalizer=None,
    ) -> EncoderInputs:
        assert normalizer is not None, "Encoder normalizer is not set!"
        if composer_normalizer is None:
            composer_normalizer = normalizer
        self._init_obs_keys(obs)
        k = self.obs_keys

        robot_id = obs.get("robot_id", None)
        if robot_id is None:
            raise ValueError(
                "robot_id must be provided in the observation for multi-robot normalization"
            )

        img_dim = obs[k["agentview"]].dim()
        assert img_dim == 5, f"Expect input to contain temporal dim, got dim={img_dim}"

        B, T = obs[k["agentview"]].shape[:2]

        # ---- make a copy; do NOT modify input dict ----
        obs_flat = dict(obs)
        for key, val in obs_flat.items():
            if torch.is_tensor(val) and val.dim() >= 2:
                obs_flat[key] = val.reshape(-1, *val.shape[2:])

        norm_keys = [
            k["eef_pos"],
            k["eef_rot"],
            k["gripper"],
            "task_embedding",
            "robot_id",
        ]
        obs_to_norm = {key: obs_flat[key] for key in norm_keys}
        obs_norm = normalizer.normalize(obs_to_norm, robot_id)
        if composer_normalizer is normalizer:
            composer_norm = obs_norm
        else:
            composer_norm = composer_normalizer.normalize(obs_to_norm, robot_id)

        agentview_image = normalize_imagenet(obs_flat[k["agentview"]], source="raw")
        if resize:
            agentview_image = resize_image(agentview_image, self.input_res)

        if self.enable_eih:
            eih_image = normalize_imagenet(obs_flat[k["in_hand"]], source="raw")
            if resize:
                eih_image = resize_image(eih_image, self.input_res)
        else:
            eih_image = None

        # convert robot_id to one-hot
        robot_id = obs_norm["robot_id"].view(-1).long()
        robot_id_one_hot = F.one_hot(robot_id, num_classes=NUM_ROBOTS).float()
        composer_robot_id = composer_norm["robot_id"].view(-1).long()
        composer_robot_id_one_hot = F.one_hot(
            composer_robot_id, num_classes=NUM_ROBOTS
        ).float()
        gripper_open = (
            torch.abs(
                composer_norm[k["gripper"]][:, 0] - composer_norm[k["gripper"]][:, 1]
            )
            - 1.0
        )

        composer_in = {
            "eef_pos": composer_norm[k["eef_pos"]],
            "eef_rot": composer_norm[k["eef_rot"]][..., :6],
            "gripper": composer_norm[k["gripper"]],
            "gripper_opening": gripper_open.unsqueeze(-1),
            "robot_id": composer_robot_id_one_hot,
            "task_embedding": composer_norm["task_embedding"],
            "raw_eef_pos": obs_flat[k["eef_pos"]].float(),
            "raw_gripper_opening": (
                torch.abs(obs_flat[k["gripper"]][:, 0] - obs_flat[k["gripper"]][:, 1])
                - 1.0
            ).float().unsqueeze(-1),
        }
        if "task_language_tokens" in obs_flat:
            composer_in["task_language_tokens"] = obs_flat[
                "task_language_tokens"
            ].float()
        self._validate_composer_in(composer_in)

        proprio = torch.cat(
            [
                obs_norm[k["eef_pos"]],
                obs_norm[k["eef_rot"]][..., :6],
                obs_norm[k["gripper"]],
            ],
            dim=-1,
        )

        return EncoderInputs(
            agentview=agentview_image,
            eye_in_hand=eih_image,
            proprio=proprio,
            robot_id=robot_id_one_hot,
            composer_in=composer_in,
            task_embedding=obs_norm["task_embedding"],
            T=T,
        )

    def _validate_composer_in(self, composer_in: dict):
        if not isinstance(composer_in, dict):
            raise TypeError(f"composer_in must be a dict, got {type(composer_in)}")

        expected = {
            "eef_pos": (3,),
            "eef_rot": (6,),
            "gripper": (2,),
            "gripper_opening": (1,),
            "robot_id": (self.num_robots,),
        }

        missing = [
            k for k in [*expected.keys(), "task_embedding"] if k not in composer_in
        ]
        if missing:
            raise ValueError(f"composer_in missing keys: {missing}")

        expected["task_embedding"] = (composer_in["task_embedding"].shape[-1],)

        for key, tail_shape in expected.items():
            x = composer_in[key]
            if x.ndim != 2 or tuple(x.shape[1:]) != tail_shape:
                raise ValueError(
                    f"composer_in['{key}'] must be [B, {tail_shape}], got {tuple(x.shape)}"
                )
