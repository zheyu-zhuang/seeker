"""Training-only Seeker policy for diffusion visual-focus pretraining."""

from copy import deepcopy

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from seeker.model.common.normalizer import MultiRobotLinearNormalizer
from seeker.model.diffusion.conditional_unet1d import ConditionalUnet1D
from seeker.model.diffusion.mask_generator import LowdimMaskGenerator
from seeker.model.seeker_training_encoder import SeekerTrainingEncoder
from seeker.policy.base_image_policy import BaseImagePolicy
from seeker.util.robomimic_env import validate_action_rep


class SeekerTrainingPolicy(BaseImagePolicy):
    """Train Seeker-based visual-focus policies with the diffusion objective."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        image_size,
        weights=None,
        stage_stride=30,
        visual_mode="agentview",
        obs_dropout=0.0,
        action_rep="absolute",
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        background_path=None,
        seeker_overrides=None,
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1

        enc_n_hidden = 128
        action_dim = action_shape[0]

        obs_feature_dim = enc_n_hidden
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = obs_feature_dim
        if obs_as_global_cond:
            input_dim = action_dim

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

        self.coarse_model = model
        self.fine_model = deepcopy(model)

        self.obs_encoder = SeekerTrainingEncoder(
            n_hidden=enc_n_hidden,
            obs_dropout=obs_dropout,
            image_size=image_size,
            visual_mode=visual_mode,
            background_path=background_path,
            weights=weights,
            seeker_config=seeker_overrides,
        )

        self.normalizer = MultiRobotLinearNormalizer()

        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.stage_stride = stage_stride

        model_num_params = sum(p.numel() for p in model.parameters()) / 1e6
        enc_num_params = sum(p.numel() for p in self.obs_encoder.parameters()) / 1e6
        action_rep = validate_action_rep(action_rep)

        self.runtime_config = {
            "Training": {
                "Objective": "Diffusion",
                "Action Chunk Representation": action_rep,
                "Visual Mode": visual_mode.replace("_", " ").title(),
                "Horizon": horizon,
                "Action Steps": n_action_steps,
                "Observation Steps": n_obs_steps,
                "Stage Stride": f"{int(self.stage_stride)} epochs",
                "Model Params": f"{model_num_params:.2f} M",
                "Encoder Params": f"{enc_num_params:.2f} M",
            },
            "Attention Seeker": self.obs_encoder.seeker.get_config(),
            "Background Randomizer": (
                self.obs_encoder.background_randomizer.get_config()
                if self.obs_encoder.background_randomizer is not None
                else {"Status": "Disabled"}
            ),
        }

    def get_runtime_config(self) -> dict:
        """Return the runtime configuration block shown at startup."""
        return self.runtime_config

    def set_normalizer(self, normalizer: MultiRobotLinearNormalizer):
        """Set action/observation normalizer for this policy and encoder."""
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.obs_encoder.set_normalizer(normalizer)

    def lambda_scheduler(self, epoch, duration, max_lambda=1.0):
        """Linear warmup scheduler in [0, max_lambda]."""
        if epoch < 0:
            return 0.0
        return min((epoch + 1) / (duration + 1e-6), max_lambda)

    def _forward_batch(self, batch, epoch_idx=None, task_instruction=None):
        """Shared preprocessing and stage selection for one training batch."""
        assert "valid_mask" not in batch

        obs = batch["obs"]
        obs_index = batch.get("obs_index", None)
        robot_id = obs.get("robot_id", None)
        task_embedding = obs.get("task_embedding", None)
        assert robot_id is not None, "robot_id must be provided"
        assert task_embedding is not None, "task_embedding must be provided"

        nactions = self.normalizer.normalize_action(batch["action"], robot_id)

        epoch_idx = self.epoch if epoch_idx is None else epoch_idx
        s = self.stage_stride
        stage = "coarse" if epoch_idx < s else "fine"
        alpha = 0.5 if epoch_idx < 3 * s else 0.6

        if epoch_idx >= 3 * s:
            self.obs_encoder.disable_random_crop = True

        feat_dict, consistency_loss = self.obs_encoder(
            obs=batch["obs"],
            stage=stage,
            overlay_alpha=alpha,
            task_instruction=task_instruction,
            obs_index=obs_index,
        )

        return nactions, feat_dict, consistency_loss, epoch_idx

    def compute_loss(
        self,
        batch,
        epoch_idx=None,
        task_instruction=None,
    ):
        """Compute diffusion loss plus stage-consistency regularization."""
        nactions, feat_dict, consistency_loss, epoch_idx = self._forward_batch(
            batch=batch,
            epoch_idx=epoch_idx,
            task_instruction=task_instruction,
        )

        batch_size = nactions.shape[0]
        trajectory = nactions

        loss = 0.0
        if feat_dict["coarse"] is not None:
            loss += self._train_step_diffusion(
                trajectory=trajectory,
                model=self.coarse_model,
                global_cond=feat_dict["coarse"].reshape(batch_size, -1),
                cond_data=trajectory,
            )
        if feat_dict["fine"] is not None:
            loss += self._train_step_diffusion(
                trajectory=trajectory,
                model=self.fine_model,
                global_cond=feat_dict["fine"].reshape(batch_size, -1),
                cond_data=trajectory,
            )
            loss /= 2.0

        s = self.stage_stride
        trimming_lambda = self.lambda_scheduler(epoch_idx - 2 * s, s // 2)
        return loss + trimming_lambda * consistency_loss

    def _train_step_diffusion(self, trajectory, model, global_cond, cond_data):
        """Single diffusion training step."""
        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        b = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (b,),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = model(
            noisy_trajectory,
            timesteps,
            local_cond=None,
            global_cond=global_cond,
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
        per_sample = reduce(loss, "b ... -> b", "mean")
        return per_sample.mean()

    def predict_action(self, obs_dict):
        """Inference is intentionally disabled for this training-only policy."""
        raise NotImplementedError("This policy is only for training and does not support inference.")
