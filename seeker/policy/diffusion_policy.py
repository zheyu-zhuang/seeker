"""Generic diffusion policy that consumes a policy observation encoder."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from einops import reduce

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from seeker.model.common.normalizer import MultiRobotLinearNormalizer
from seeker.model.diffusion.conditional_unet1d import ConditionalUnet1D
from seeker.model.diffusion.mask_generator import LowdimMaskGenerator
from seeker.policy.base_image_policy import BaseImagePolicy

from seeker.dataset.action import validate_action_rep


class DiffusionPolicy(BaseImagePolicy):
    """Action diffusion policy conditioned on encoded observations."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        obs_encoder: Optional[dict] = None,
        action_rep="absolute",
        enc_n_hidden=128,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        # extra kwargs passed to scheduler.step during sampling
        **kwargs,
    ):
        super().__init__()

        # Parse action/observation shapes.
        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) != 1:
            raise ValueError(f"Expected 1D action shape, got {action_shape}")
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            type = attr.get("type", "low_dim")
            if type == "rgb":
                obs_config["rgb"].append(key)
            elif type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        if obs_encoder is None:
            raise ValueError("DiffusionPolicy requires an obs_encoder config or module")
        if not isinstance(obs_encoder, nn.Module):
            obs_encoder = hydra.utils.instantiate(obs_encoder)

        # Build diffusion model.
        obs_feature_dim = enc_n_hidden * n_obs_steps
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        # Action normalizer for policy targets/predictions. Observation
        # normalization is delegated to the configured observation encoder.
        self.local_normalizer = MultiRobotLinearNormalizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        diff_num_params = sum(p.numel() for p in model.parameters()) / 1e6
        enc_num_params = sum(p.numel() for p in self.obs_encoder.parameters()) / 1e6
        action_rep = validate_action_rep(action_rep)

        self.runtime_config = {
            "Training": {
                "Objective": "Diffusion",
                "Action Chunk Rep": action_rep,
                "Action Steps": n_action_steps,
                "Observation Steps": n_obs_steps,
                "Model Params": f"{diff_num_params:.1f} M",
                "Encoder Params": f"{enc_num_params:.1f} M",
            },
            **self.obs_encoder.get_config(),
        }

    def get_runtime_config(self) -> Dict:
        """Return the runtime configuration block shown at startup."""
        return self.runtime_config

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # kwargs forwarded to scheduler.step
        **kwargs,
    ):
        """Run reverse diffusion with hard conditioning."""
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )

            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def set_normalizer(self, normalizer: MultiRobotLinearNormalizer):
        """Set action normalizer state for this policy."""
        self.local_normalizer.load_state_dict(normalizer.state_dict())
        self.obs_encoder.set_normalizer(normalizer)

    def compute_loss(
        self,
        batch,
        epoch_idx=None,
    ):
        """Compute diffusion training loss for one batch."""
        obs = batch["obs"]
        if "robot_id" not in obs:
            raise KeyError("batch['obs'] must include robot_id")
        # robot_id is [B, T], action is [B, H, Da]; use first step for normalization.
        robot_id = obs["robot_id"][:, 0:1]
        nactions = self.local_normalizer.normalize_action(batch["action"], robot_id)
        batch_size = nactions.shape[0]
        trajectory = nactions
        cond_data = trajectory

        local_cond = None
        obs_index = batch.get("obs_index", None)

        # Flatten obs indices for encoder cache interface.
        if obs_index is not None:
            obs_index = obs_index.reshape(-1)

        if "oracle_info" in batch:
            obs_feature = self.obs_encoder(
                obs=obs,
                obs_index=obs_index,
                oracle_info=batch["oracle_info"],
            )
        else:
            obs_feature = self.obs_encoder(obs=obs, obs_index=obs_index)
        # Reshape back to [B, Do].
        global_cond = obs_feature.reshape(batch_size, -1)
        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask

        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        model = self.model
        pred = model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Sample actions from observations via conditional diffusion."""
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action inference is not implemented")
        if "robot_id" not in obs_dict:
            raise KeyError("obs_dict must include robot_id")
        device = self.device

        # robot_id is [B, T], action is [B, H, Da]; use first step for normalization.
        robot_id = obs_dict["robot_id"][:, 0:1]

        value = next(iter(obs_dict.values()))
        B, _ = value.shape[:2]

        T = self.horizon
        Da = self.action_dim

        dtype = self.dtype

        local_cond = None
        global_cond = None

        if "oracle_info" in obs_dict:
            obs_feature = self.obs_encoder(
                obs_dict,
                oracle_info=obs_dict["oracle_info"],
            )
        else:
            obs_feature = self.obs_encoder(obs_dict)

        global_cond = obs_feature.reshape(B, -1)
        # Empty action tensor + mask (no hard conditioning here).
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Run reverse diffusion sampling.
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # Unnormalize predictions.
        naction_pred = nsample[..., :Da]
        action_pred = self.local_normalizer.unnormalize_action(naction_pred, robot_id)

        action = action_pred[:, : self.n_action_steps]

        result = {"action": action, "action_pred": action_pred}
        return result
