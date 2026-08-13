if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb

from seeker.util.json_logger import JsonLogger
from seeker.util.torch_utils import dict_apply, optimizer_to
from seeker.model.common.lr_scheduler import get_scheduler
from seeker.model.diffusion.ema_model import EMAModel
from seeker.dataset.sampler import build_task_balanced_sampler

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from seeker.workspace.base_workspace import BaseWorkspace
from seeker.policy.base_image_policy import BaseImagePolicy
from seeker.util.formatting import fold_path_from_marker, pretty_print_nested


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainSeekerWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: BaseImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
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
        train_dataset.set_mode("train")
        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        sampler_cfg = dataloader_cfg.pop("sampler", None)
        train_sampler = build_task_balanced_sampler(
            train_dataset,
            sampler_cfg,
            seed=cfg.training.seed,
        )
        if train_sampler is not None:
            dataloader_cfg.pop("shuffle", None)
            train_dataloader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                shuffle=False,
                **dataloader_cfg,
            )
        else:
            train_dataloader = DataLoader(train_dataset, **dataloader_cfg)
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

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        viz_dir = os.path.join(self.output_dir, "visualization")
        os.makedirs(viz_dir, exist_ok=True)

        self.model.obs_encoder.initialize_internal(
            dataset_size=train_dataset.n_samples_active,
            device=device,
            viz_dir=viz_dir,
        )


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

                        try:
                            task_instruction = batch["task_instruction"].copy()
                            del batch["task_instruction"]
                        except KeyError:
                            task_instruction = None

                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(
                            batch,
                            task_instruction=task_instruction,
                            epoch_idx=self.epoch,
                        )
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
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    seeker_light_path = cfg.checkpoint.get("seeker_light_path", None)
                    if seeker_light_path is not None:
                        seeker_weights = self.model.obs_encoder.seeker.state_dict()
                        attent_seeker_key = "obs_encoder.seeker"
                        seeker_weights = {
                            k.replace(f"{attent_seeker_key}.", ""): v
                            for k, v in seeker_weights.items()
                        }
                        seeker_light_dir = os.path.dirname(seeker_light_path)
                        if seeker_light_dir:
                            os.makedirs(seeker_light_dir, exist_ok=True)
                        torch.save(seeker_weights, seeker_light_path)

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
