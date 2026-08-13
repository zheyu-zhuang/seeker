"""Train a MVT heatmap predictor on RVT2Heatmap heuristic labels."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from seeker.dataset.rvt2_heatmap_dataset import (
    RVT2_HEATMAP_SHAPE_META,
    RVT2HeatmapPatchDataset,
    build_rvt2_heatmap_patch_records,
    gripper_stats,
    resolve_dino_checkpoint,
    resolve_rvt2_heatmap_dataset_path,
    split_patch_records,
)
from seeker.dataset.sampler import build_task_balanced_sampler
from seeker.dataset.mimicgen_dataset import MimicGenDataset
from seeker.model.rvt2_heatmap import (
    PatchFeatureBackbone,
    PatchActivationHead,
)
from seeker.util.image_ops import image_to_float01, normalize_imagenet, resize_image
from seeker.util.visual_randomizer import BackgroundRandomizer
from seeker.focus_toolbox.viz import draw_patch_debug
from seeker.focus_toolbox.config import (
    load_rvt2_heatmap_config,
    load_rvt2_query_config,
)
from seeker.util.task_meta import NUM_ROBOTS
from seeker.workspace.base_workspace import BaseWorkspace


def _gaussian_patch_targets(
    target: torch.Tensor,
    *,
    grid_size: int,
    sigma: float,
) -> torch.Tensor:
    """Build soft patch heatmap labels centered on the target patch."""
    if sigma <= 0:
        raise ValueError("sigma must be positive for Gaussian targets")
    device = target.device
    coords = torch.arange(grid_size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    patch_y = (target // grid_size).to(torch.float32)[:, None, None]
    patch_x = (target % grid_size).to(torch.float32)[:, None, None]
    dist2 = (yy[None] - patch_y).square() + (xx[None] - patch_x).square()
    heatmap = torch.exp(-0.5 * dist2 / float(sigma * sigma))
    heatmap = heatmap.flatten(1)
    return heatmap / heatmap.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _patch_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    grid_size: int,
    sigma: float,
) -> torch.Tensor:
    if sigma <= 0:
        return F.cross_entropy(logits, target)
    soft_target = _gaussian_patch_targets(
        target,
        grid_size=grid_size,
        sigma=sigma,
    )
    return -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _save_visualization_grid(
    *,
    patch_backbone: PatchFeatureBackbone,
    head: PatchActivationHead,
    background_randomizer: Optional[BackgroundRandomizer],
    overlay_alpha: Optional[float],
    loader: DataLoader,
    device: torch.device,
    dino_image_size: int,
    patch_size: int,
    num_samples: int,
    alpha: float,
    output_path: Path,
) -> None:
    if num_samples <= 0:
        return
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    head.eval()
    patch_backbone.eval()
    with torch.no_grad():
        image = batch["image"].to(device, non_blocking=True)
        image_norm = resize_image(
            normalize_imagenet(image, source="raw"),
            dino_image_size,
        )
        image_norm = _random_overlay_like_seeker(
            image_norm,
            background_randomizer=background_randomizer,
            overlay_alpha=overlay_alpha,
        )
        image_vis = image_to_float01(image_norm, source="imagenet")
        target = batch["target_patch"].to(device, non_blocking=True)
        logits = _head_logits(
            head=head,
            patches=patch_backbone(image_norm),
            batch=batch,
            device=device,
        )
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

    grid_size = int(dino_image_size) // int(patch_size)
    n = min(int(num_samples), int(image.shape[0]))
    panels = []
    for i in range(n):
        prob_grid = probs[i].detach().cpu().reshape(grid_size, grid_size)
        panels.append(
            draw_patch_debug(
                image=image_vis[i].detach().cpu(),
                prob_grid=prob_grid,
                target_patch=int(target[i].detach().cpu()),
                pred_patch=int(pred[i].detach().cpu()),
                source_frame=int(batch["source_frame"][i]),
                target_frame=int(batch["target_frame"][i]),
                alpha=alpha,
            )
        )
    if not panels:
        return

    cols = min(4, len(panels))
    rows = int(math.ceil(len(panels) / cols))
    canvas = Image.new(
        "RGB",
        (cols * dino_image_size, rows * dino_image_size),
        color=(255, 255, 255),
    )
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % cols) * dino_image_size, (i // cols) * dino_image_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _make_visualization_loader(
    dataset: Dataset,
    *,
    num_samples: int,
    seed: int,
    epoch: int,
    mode: str,
) -> Optional[DataLoader]:
    n = len(dataset)
    if n == 0 or num_samples <= 0:
        return None
    k = min(int(num_samples), n)
    mode = str(mode).strip().lower()
    if mode == "uniform":
        indices = np.linspace(0, n - 1, num=k, dtype=np.int64).tolist()
    elif mode == "random":
        rng = np.random.default_rng(int(seed) + 1009 * int(epoch))
        indices = rng.choice(n, size=k, replace=False).astype(np.int64).tolist()
    else:
        raise ValueError(f"Unknown visualization sampling mode: {mode!r}")
    return DataLoader(
        Subset(dataset, indices),
        batch_size=k,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _head_logits(
    *,
    head: PatchActivationHead,
    patches: torch.Tensor,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    eef_pos = batch.get("eef_pos")
    task_language_tokens = batch.get("task_language_tokens")
    return head(
        patches,
        batch["gripper"].to(device, non_blocking=True),
        batch["task_embedding"].to(device, non_blocking=True),
        batch["robot_id"].to(device, non_blocking=True),
        eef_pos=None if eef_pos is None else eef_pos.to(device, non_blocking=True),
        task_language_tokens=(
            None
            if task_language_tokens is None
            else task_language_tokens.to(device, non_blocking=True)
        ),
    )


def _run_epoch(
    *,
    patch_backbone: PatchFeatureBackbone,
    head: PatchActivationHead,
    background_randomizer: Optional[BackgroundRandomizer],
    overlay_alpha: Optional[float],
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    dino_image_size: int,
    patch_size: int,
    target_sigma_patches: float,
    max_steps: Optional[int],
    desc: str,
    show_progress: bool,
) -> dict:
    train = optimizer is not None
    patch_backbone.train(train and patch_backbone.backbone_type != "dino")
    head.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_top5 = 0.0
    total_count = 0
    grid_size = int(dino_image_size) // int(patch_size)

    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            desc=desc,
            leave=True,
            file=sys.stdout,
            dynamic_ncols=True,
        )
    for step, batch in enumerate(iterator, start=1):
        image = batch["image"].to(device, non_blocking=True)
        image = resize_image(
            normalize_imagenet(image, source="raw"),
            dino_image_size,
        )
        if train:
            image = _random_overlay_like_seeker(
                image,
                background_randomizer=background_randomizer,
                overlay_alpha=overlay_alpha,
            )
        target = batch["target_patch"].to(device, non_blocking=True)

        patches = patch_backbone(image)
        logits = _head_logits(head=head, patches=patches, batch=batch, device=device)
        loss = _patch_heatmap_loss(
            logits,
            target,
            grid_size=grid_size,
            sigma=float(target_sigma_patches),
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            topk = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
            count = int(target.shape[0])
            total_loss += float(loss.item()) * count
            total_acc += float((pred == target).float().sum().item())
            total_top5 += float((topk == target[:, None]).any(dim=1).float().sum().item())
            total_count += count
            if show_progress:
                iterator.set_postfix(
                    loss=total_loss / max(total_count, 1),
                    acc=total_acc / max(total_count, 1),
                )

        if max_steps is not None and step >= max_steps:
            break

    if total_count == 0:
        return {"loss": None, "acc": None, "top5": None, "n": 0}
    return {
        "loss": total_loss / total_count,
        "acc": total_acc / total_count,
        "top5": total_top5 / total_count,
        "n": total_count,
    }


def _save_checkpoint(
    path: Path,
    *,
    head: PatchActivationHead,
    optimizer: torch.optim.Optimizer,
    patch_backbone: PatchFeatureBackbone,
    training_config: dict,
    rvt2_heatmap_config: dict,
    head_config: dict,
    gripper_mean: np.ndarray,
    gripper_std: np.ndarray,
    label_stats: dict,
    epoch: int,
    metrics: dict,
) -> None:
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "patch_backbone_state_dict": patch_backbone.state_dict(),
            "patch_backbone": patch_backbone.backbone_type,
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": training_config,
            "rvt2_heatmap_config": rvt2_heatmap_config,
            "head_config": head_config,
            "gripper_mean": gripper_mean,
            "gripper_std": gripper_std,
            "label_stats": label_stats,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def _load_checkpoint(
    path: str,
    *,
    head: PatchActivationHead,
    patch_backbone: PatchFeatureBackbone,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    load_optimizer: bool,
) -> int:
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"RVT2Heatmap checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    if not isinstance(payload, dict) or "head_state_dict" not in payload:
        raise ValueError(f"Invalid RVT2Heatmap checkpoint: {ckpt_path}")

    head.load_state_dict(payload["head_state_dict"], strict=True)
    if "patch_backbone_state_dict" not in payload:
        raise ValueError("RVT2Heatmap checkpoint is missing patch_backbone_state_dict.")
    patch_backbone.load_state_dict(payload["patch_backbone_state_dict"], strict=True)
    if load_optimizer and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if "epoch" not in payload:
        raise ValueError("RVT2Heatmap checkpoint is missing epoch.")
    return int(payload["epoch"]) + 1


def _seeker_overlay_alpha(epoch: int, stage_stride: float) -> float:
    return 0.5 if int(epoch) < 3 * float(stage_stride) else 0.6


def _random_overlay_like_seeker(
    image: torch.Tensor,
    *,
    background_randomizer: Optional[BackgroundRandomizer],
    overlay_alpha: Optional[float],
) -> torch.Tensor:
    if background_randomizer is None or overlay_alpha is None:
        return image
    B = int(image.shape[0])
    count = int(B * 0.5)
    if count <= 0:
        return image
    bg = background_randomizer(B).to(device=image.device, dtype=image.dtype)
    if bg.shape[-2:] != image.shape[-2:]:
        bg = F.interpolate(bg, size=image.shape[-2:], mode="bilinear", align_corners=False)
    idx = torch.randperm(B, device=image.device)[:count]
    out = image.clone()
    alpha = float(overlay_alpha)
    out[idx] = image[idx] * alpha + bg[idx] * (1.0 - alpha)
    return out


def _build_background_randomizer(args: SimpleNamespace, rvt2_heatmap_cfg: dict):
    cfg = getattr(args, "background_overlay", None)
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return None
    background_path = cfg.get("background_path")
    if not background_path or not os.path.exists(str(background_path)):
        raise FileNotFoundError(
            "background_overlay.background_path must exist when RVT2 heatmap "
            f"pretraining overlay is enabled, got {background_path!r}"
        )
    image_size = int(rvt2_heatmap_cfg["dino_image_size"])
    return BackgroundRandomizer(
        input_shape=(image_size, image_size),
        background_path=str(background_path),
    )


def _prepare_runtime(cfg: OmegaConf, output_dir: str) -> dict:
    training_config = OmegaConf.to_container(cfg, resolve=True)
    args = SimpleNamespace(**training_config)
    rvt2_heatmap_overrides = {}
    if isinstance(getattr(args, "rvt2_heatmap", None), dict):
        rvt2_heatmap_overrides["rvt2_heatmap"] = args.rvt2_heatmap
    if isinstance(getattr(args, "query_composer", None), dict):
        rvt2_heatmap_overrides["query_composer"] = args.query_composer
    rvt2_heatmap_cfg = load_rvt2_heatmap_config(
        overrides=rvt2_heatmap_overrides or None
    )
    query_cfg = load_rvt2_query_config(overrides=rvt2_heatmap_overrides or None)
    rvt2_heatmap_cfg_dict = dict(rvt2_heatmap_cfg)
    if args.resume and args.load_weights:
        raise ValueError("Use only one of --resume or --load-weights.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = resolve_rvt2_heatmap_dataset_path(args.task_name, args.dataset_path)
    dino_ckpt = (
        resolve_dino_checkpoint(rvt2_heatmap_cfg["dino_ckpt"])
        if rvt2_heatmap_cfg["patch_backbone"] == "dino"
        else None
    )
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    background_randomizer = _build_background_randomizer(args, rvt2_heatmap_cfg)
    return {
        "training_config": training_config,
        "args": args,
        "rvt2_heatmap_cfg": rvt2_heatmap_cfg,
        "rvt2_heatmap_cfg_dict": rvt2_heatmap_cfg_dict,
        "query_cfg": query_cfg,
        "dataset_path": dataset_path,
        "dino_ckpt": dino_ckpt,
        "background_randomizer": background_randomizer,
        "output_dir": output_path,
    }


def _prepare_training_data(runtime: dict) -> dict:
    args = runtime["args"]
    rvt2_heatmap_cfg = runtime["rvt2_heatmap_cfg"]
    active_demo_count = None
    if args.n_demo is not None:
        active_demo_count = int(args.n_demo) + int(args.skip_first_episodes)

    dataset = MimicGenDataset(
        shape_meta=RVT2_HEATMAP_SHAPE_META,
        dataset_path=str(runtime["dataset_path"]),
        image_size=None,
        horizon=1,
        val_ratio=0.0,
        n_demo=active_demo_count,
        demo_count_mode=getattr(args, "dataset_demo_count_mode", "total"),
        action_rep="absolute",
        cache_dir=args.cache_dir,
        include_oracle_info=True,
    )
    start_episode = min(int(args.skip_first_episodes), dataset.n_demo_active)
    episode_indices = list(range(start_episode, dataset.n_demo_active))
    if not episode_indices:
        raise ValueError("No episodes selected after --skip-first-episodes.")

    records, label_stats = build_rvt2_heatmap_patch_records(
        dataset,
        episode_indices=episode_indices,
        camera=rvt2_heatmap_cfg["camera"],
        dino_image_size=rvt2_heatmap_cfg["dino_image_size"],
        patch_size=rvt2_heatmap_cfg["patch_size"],
        joint_vel_atol=rvt2_heatmap_cfg["joint_vel_atol"],
        stopped_buffer_len=rvt2_heatmap_cfg["stopped_buffer_len"],
        include_final=rvt2_heatmap_cfg["include_final"],
        mute_initial_gripper_open=rvt2_heatmap_cfg["mute_initial_gripper_open"],
        show_progress=args.show_label_progress,
    )
    if not records:
        raise ValueError(f"No trainable labels were built. Label stats: {label_stats}")

    train_records, val_records = split_patch_records(
        records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    gripper_mean, gripper_std = gripper_stats(train_records)

    train_set = RVT2HeatmapPatchDataset(dataset, train_records, gripper_mean, gripper_std)
    val_set = RVT2HeatmapPatchDataset(dataset, val_records, gripper_mean, gripper_std)
    train_sampler = build_task_balanced_sampler(
        train_set,
        getattr(args, "sampler", None),
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0 and len(val_set) > 0,
    )
    return {
        "dataset": dataset,
        "train_set": train_set,
        "val_set": val_set,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_sampler": train_sampler,
        "records": records,
        "train_records": train_records,
        "val_records": val_records,
        "label_stats": label_stats,
        "gripper_mean": gripper_mean,
        "gripper_std": gripper_std,
    }


def _prepare_model_state(runtime: dict) -> dict:
    args = runtime["args"]
    rvt2_heatmap_cfg = runtime["rvt2_heatmap_cfg"]
    query_cfg = runtime["query_cfg"]
    device = torch.device(args.device)
    patch_backbone = PatchFeatureBackbone(
        backbone_type=rvt2_heatmap_cfg["patch_backbone"],
        dino_ckpt_path=runtime["dino_ckpt"],
        image_size=rvt2_heatmap_cfg["dino_image_size"],
        patch_size=rvt2_heatmap_cfg["patch_size"],
        conv_dim=rvt2_heatmap_cfg["conv_patch_dim"],
    ).to(device)
    patch_dim = int(patch_backbone.output_dim)
    head_config = {
        "patch_dim": patch_dim,
        "gripper_dim": 1,
        "hidden_dim": int(rvt2_heatmap_cfg["hidden_dim"]),
        "grid_size": int(rvt2_heatmap_cfg["dino_image_size"])
        // int(rvt2_heatmap_cfg["patch_size"]),
        "task_emb_dim": int(query_cfg["task_emb_dim"]),
        "num_robots": int(NUM_ROBOTS),
        "query_hidden_mult": int(query_cfg["hidden_mult"]),
        "proprio_mode": str(query_cfg["proprio_mode"]),
        "proprio_dim": int(query_cfg["proprio_dim"]),
        "language_seq_len": 77,
        "transformer_depth": int(rvt2_heatmap_cfg["transformer_depth"]),
        "transformer_heads": int(rvt2_heatmap_cfg["transformer_heads"]),
        "transformer_dropout": float(rvt2_heatmap_cfg["transformer_dropout"]),
    }
    head = PatchActivationHead(**head_config).to(device)
    trainable_params = list(head.parameters())
    if patch_backbone.backbone_type != "dino":
        trainable_params += list(patch_backbone.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    return {
        "patch_backbone": patch_backbone,
        "head": head,
        "optimizer": optimizer,
        "head_config": head_config,
        "device": device,
    }


def _restore_training_state(runtime: dict, model_state: dict) -> int:
    args = runtime["args"]
    if args.resume:
        return _load_checkpoint(
            args.resume,
            head=model_state["head"],
            patch_backbone=model_state["patch_backbone"],
            optimizer=model_state["optimizer"],
            device=model_state["device"],
            load_optimizer=not args.no_resume_optimizer,
        )
    if args.load_weights:
        _load_checkpoint(
            args.load_weights,
            head=model_state["head"],
            patch_backbone=model_state["patch_backbone"],
            optimizer=model_state["optimizer"],
            device=model_state["device"],
            load_optimizer=False,
        )
    return 1


def _write_run_summary(
    runtime: dict,
    data: dict,
    model_state: dict,
    start_epoch: int,
) -> None:
    args = runtime["args"]
    run_summary = {
        "dataset_path": str(runtime["dataset_path"]),
        "cache_dir": data["dataset"].cache_dir,
        "dino_ckpt": (
            None if runtime["dino_ckpt"] is None else str(runtime["dino_ckpt"])
        ),
        "patch_backbone": runtime["rvt2_heatmap_cfg"]["patch_backbone"],
        "output_dir": str(runtime["output_dir"]),
        "records": len(data["records"]),
        "train_records": len(data["train_records"]),
        "val_records": len(data["val_records"]),
        "label_stats": data["label_stats"],
        "resume": args.resume,
        "load_weights": args.load_weights,
        "start_epoch": start_epoch,
        "rvt2_heatmap_config": runtime["rvt2_heatmap_cfg_dict"],
        "head_config": model_state["head_config"],
    }
    with (runtime["output_dir"] / "run_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_summary, f, indent=2)
        f.write("\n")
    if args.print_run_summary:
        print(json.dumps(run_summary, indent=2), flush=True)


def _run_training_loop(
    *,
    runtime: dict,
    data: dict,
    model_state: dict,
    start_epoch: int,
    wandb_run=None,
) -> None:
    args = runtime["args"]
    rvt2_heatmap_cfg = runtime["rvt2_heatmap_cfg"]
    log_path = runtime["output_dir"] / "train_log.jsonl"
    checkpoint_every = int(getattr(args, "checkpoint_every", 0) or 0)
    for epoch in range(start_epoch, int(args.epochs) + 1):
        overlay_alpha = _seeker_overlay_alpha(epoch, args.stage_stride)
        if data["train_sampler"] is not None:
            data["train_sampler"].set_epoch(epoch)
        train_metrics = _run_epoch(
            patch_backbone=model_state["patch_backbone"],
            head=model_state["head"],
            background_randomizer=runtime["background_randomizer"],
            overlay_alpha=overlay_alpha,
            loader=data["train_loader"],
            optimizer=model_state["optimizer"],
            device=model_state["device"],
            dino_image_size=rvt2_heatmap_cfg["dino_image_size"],
            patch_size=rvt2_heatmap_cfg["patch_size"],
            target_sigma_patches=rvt2_heatmap_cfg["target_sigma_patches"],
            max_steps=args.max_train_steps,
            desc=f"epoch {epoch}/{args.epochs}",
            show_progress=True,
        )
        if len(data["val_set"]) > 0:
            val_metrics = _run_epoch(
                patch_backbone=model_state["patch_backbone"],
                head=model_state["head"],
                background_randomizer=runtime["background_randomizer"],
                overlay_alpha=None,
                loader=data["val_loader"],
                optimizer=None,
                device=model_state["device"],
                dino_image_size=rvt2_heatmap_cfg["dino_image_size"],
                patch_size=rvt2_heatmap_cfg["patch_size"],
                target_sigma_patches=rvt2_heatmap_cfg["target_sigma_patches"],
                max_steps=args.max_val_steps,
                desc=f"epoch {epoch} val",
                show_progress=False,
            )
        else:
            val_metrics = {"loss": None, "acc": None, "top5": None, "n": 0}

        payload = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        if int(args.vis_every_epochs) > 0 and epoch % int(args.vis_every_epochs) == 0:
            split_name = "val" if len(data["val_set"]) > 0 else "train"
            vis_set = (
                data["val_set"] if len(data["val_set"]) > 0 else data["train_set"]
            )
            vis_loader = _make_visualization_loader(
                vis_set,
                num_samples=args.vis_num_samples,
                seed=args.seed,
                epoch=epoch,
                mode=args.vis_sampling,
            )
            vis_path = (
                runtime["output_dir"]
                / "visualizations"
                / f"epoch_{epoch:04d}_{split_name}.png"
            )
            if vis_loader is not None:
                _save_visualization_grid(
                    patch_backbone=model_state["patch_backbone"],
                    head=model_state["head"],
                    background_randomizer=runtime["background_randomizer"],
                    overlay_alpha=overlay_alpha,
                    loader=vis_loader,
                    device=model_state["device"],
                    dino_image_size=rvt2_heatmap_cfg["dino_image_size"],
                    patch_size=rvt2_heatmap_cfg["patch_size"],
                    num_samples=args.vis_num_samples,
                    alpha=args.vis_alpha,
                    output_path=vis_path,
                )
                payload["visualization"] = str(vis_path)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        if args.print_epoch_metrics:
            print(json.dumps(payload), flush=True)
        if wandb_run is not None:
            step_log = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_top5": train_metrics["top5"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_top5": val_metrics["top5"],
            }
            wandb_run.log(step_log, step=epoch)

        should_checkpoint = (
            checkpoint_every > 0
            and ((epoch % checkpoint_every) == 0 or epoch == int(args.epochs))
        )
        if should_checkpoint:
            _save_checkpoint(
                runtime["output_dir"] / "latest.pt",
                head=model_state["head"],
                patch_backbone=model_state["patch_backbone"],
                optimizer=model_state["optimizer"],
                training_config=runtime["training_config"],
                rvt2_heatmap_config=runtime["rvt2_heatmap_cfg_dict"],
                head_config=model_state["head_config"],
                gripper_mean=data["gripper_mean"],
                gripper_std=data["gripper_std"],
                label_stats=data["label_stats"],
                epoch=epoch,
                metrics=payload,
            )


class TrainRVT2HeatmapWorkspace(BaseWorkspace):
    """Hydra workspace for RVT2Heatmap training."""

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.runtime = _prepare_runtime(cfg, self.output_dir)
        self.model_state = _prepare_model_state(self.runtime)
        self.start_epoch = 1
        self.data = None

    def run(self):
        wandb_run = None
        if "logging" in self.cfg:
            os.environ.setdefault("WANDB_SILENT", "true")
            wandb_run = wandb.init(
                dir=str(self.runtime["output_dir"]),
                config=OmegaConf.to_container(self.cfg, resolve=True),
                settings=wandb.Settings(console="off"),
                **self.cfg.logging,
            )
            wandb.config.update({"output_dir": str(self.runtime["output_dir"])})

        self.data = _prepare_training_data(self.runtime)
        self.start_epoch = _restore_training_state(self.runtime, self.model_state)
        _write_run_summary(
            self.runtime,
            self.data,
            self.model_state,
            self.start_epoch,
        )
        _run_training_loop(
            runtime=self.runtime,
            data=self.data,
            model_state=self.model_state,
            start_epoch=self.start_epoch,
            wandb_run=wandb_run,
        )
