import copy
import os
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb
from torch.utils._pytree import tree_map

from seeker.workspace.training_io import JsonLogger, TopKCheckpointManager
from seeker.model.common.lr_scheduler import get_scheduler
from seeker.model.diffusion.ema_model import EMAModel
from seeker.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from einops import rearrange

from seeker.policy.diffusion_policy import DiffusionPolicy
from seeker.util.formatting import fold_path_from_marker, pretty_print_nested
from seeker.util.mirror_augmentation import MirrorObsActionAugmentor


class TrainFocusPolicyWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionPolicy = hydra.utils.instantiate(
            cfg.policy
        )

        self.ema_model: DiffusionPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

    @staticmethod
    def _validate_mirror_aug_baseline_only(cfg: OmegaConf, enabled: bool):
        if not enabled:
            return
        try:
            views = cfg.policy.obs_encoder.focus_view_transform.views
        except Exception:
            return
        baseline_modes = {"pass_through", "random_overlay", "disabled"}
        active_modes = {str(mode) for mode in views.values()}
        unsupported = sorted(active_modes - baseline_modes)
        if unsupported:
            raise ValueError(
                "mirror_augmentation is baseline-only for now; disable Seeker focus "
                f"modes before enabling it. Unsupported modes: {unsupported}"
            )

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        resume_status = "From Scratch"

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                resume_status = fold_path_from_marker(lastest_ckpt_path)
                self.load_checkpoint(path=lastest_ckpt_path)
                self.epoch += 1
                self.global_step += 1

        runtime_cfg = copy.deepcopy(self.model.get_runtime_config())
        runtime_cfg["Training"]["Checkpoint"] = resume_status
        pretty_print_nested(
            runtime_cfg,
            title="Runtime Configuration",
            pad_before=True,
            pad_after=True,
        )

        # configure dataset
        train_dataset = hydra.utils.instantiate(cfg.task.dataset)
        mirror_augmentor = MirrorObsActionAugmentor(
            shape_meta=OmegaConf.to_container(cfg.shape_meta, resolve=True),
            action_rep=cfg.action_rep,
            config=OmegaConf.to_container(
                cfg.get("mirror_augmentation", None), resolve=True
            ),
        )
        self._validate_mirror_aug_baseline_only(cfg, mirror_augmentor.enabled)
        train_dataset.set_mode("train")
        train_dataloader = DataLoader(train_dataset, **cfg.dataloader)
        val_dataset = hydra.utils.instantiate(cfg.task.dataset)
        val_dataset.set_mode("eval")
        val_dataloader = None
        if len(val_dataset) > 0:
            val_dataloader_cfg = copy.deepcopy(cfg.val_dataloader)
            val_dataloader_cfg["shuffle"] = True
            val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)
        normalizer = train_dataset.get_normalizer()

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir
        )

        # configure logging
        os.environ.setdefault("WANDB_SILENT", "true")
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(console="off"),
            **cfg.logging,
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # device transfer
        if not torch.cuda.is_available():
            raise RuntimeError("training requires a CUDA GPU")
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=device)

        self.model.obs_encoder.initialize_internal(
            dataset_size=train_dataset.n_samples_active,
            device=device,
            viz_dir=os.path.join(self.output_dir, "visualization"),
        )
        if self.ema_model is not None:
            self.ema_model.obs_encoder.initialize_internal(
                dataset_size=train_dataset.n_samples_active,
                device=device,
                viz_dir=os.path.join(self.output_dir, "visualization"),
            )

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            while self.epoch < cfg.training.num_epochs:
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = tree_map(
                            lambda x: x.to(device, non_blocking=True)
                            if torch.is_tensor(x)
                            else x,
                            batch,
                        )
                        mirror_augmentor(batch)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch, epoch_idx=self.epoch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if (
                            self.global_step % cfg.training.gradient_accumulate_every
                            == 0
                        ):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                val_every = int(cfg.training.val_every)
                should_validate = (
                    val_dataloader is not None
                    and val_every > 0
                    and (self.epoch % val_every) == 0
                )
                if should_validate:
                    val_losses = []
                    obs_encoder = getattr(policy, "obs_encoder", None)
                    if hasattr(obs_encoder, "request_visualization"):
                        obs_encoder.request_visualization(
                            split="val",
                            step=self.global_step,
                        )
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = tree_map(
                                lambda x: x.to(device, non_blocking=True)
                                if torch.is_tensor(x)
                                else x,
                                batch,
                            )
                            mirror_augmentor.center_batch(batch)
                            loss = policy.compute_loss(batch, epoch_idx=self.epoch)
                            val_losses.append(loss.item())
                            if (
                                cfg.training.max_val_steps is not None
                                and batch_idx >= cfg.training.max_val_steps - 1
                            ):
                                break
                    if len(val_losses) > 0:
                        step_log["val_loss"] = float(np.mean(val_losses))
                # run rollout
                did_rollout = (
                    self.epoch % cfg.training.rollout_every
                ) == 0 and self.epoch != 0
                if did_rollout:
                    runner_log = env_runner.run(policy, epoch=self.epoch)
                    step_log.update(runner_log)

                # Log as soon as epoch metrics are ready. Checkpointing can be slow,
                # so logging after it makes rollout media appear one cycle late.
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)

                # checkpoint
                is_last_epoch = self.epoch == (cfg.training.num_epochs - 1)
                checkpoint_every = int(cfg.training.get("checkpoint_every", 0))
                should_checkpoint = (
                    checkpoint_every > 0
                    and (
                        (
                            self.epoch % checkpoint_every
                        ) == 0 and self.epoch != 0
                        or is_last_epoch
                    )
                )
                if should_checkpoint:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    if topk_manager.monitor_key in metric_dict:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            if cfg.checkpoint.get("save_topk_full_ckpt", False):
                                self.save_checkpoint(path=topk_ckpt_path)
                            else:
                                policy_key = (
                                    "ema_model" if cfg.training.use_ema else "model"
                                )
                                self.save_state_dict_checkpoint(
                                    path=topk_ckpt_path,
                                    state_dicts={policy_key: policy},
                                    metadata={
                                        "lightweight": True,
                                        "checkpoint_type": "topk_policy",
                                        "policy_key": policy_key,
                                        "epoch": int(self.epoch),
                                        "global_step": int(self.global_step),
                                    },
                                )
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                self.global_step += 1
                self.epoch += 1
