"""Train command for Seeker CLI."""

from __future__ import annotations

import argparse
import sys

import hydra
import torch
from omegaconf import OmegaConf

from ..workspace.base_workspace import BaseWorkspace


MAX_STEPS = {
    "square_d0": 400,
    "stack_d1": 400,
    "stack_three_d1": 400,
    "square_d2": 400,
    "threading_d2": 400,
    "coffee_d2": 400,
    "three_piece_assembly_d2": 500,
    "hammer_cleanup_d1": 500,
    "mug_cleanup_d1": 500,
    "kitchen_d1": 800,
    "nut_assembly_d0": 500,
    "pick_place_d0": 1000,
    "coffee_preparation_d1": 800,
    "tool_hang": 700,
    "can": 400,
    "lift": 400,
    "square": 400,
}


def _register_resolvers() -> None:
    resolvers = {
        "eval": eval,
        "add_int": lambda a, b: int(a) + int(b),
        "divide": lambda a, b: float(a) / float(b),
        "get_max_steps": lambda task_name: MAX_STEPS.get(task_name, 800),
    }
    for name, resolver in resolvers.items():
        OmegaConf.register_new_resolver(name, resolver, replace=True)


def _enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


@hydra.main(
    version_base=None,
    config_path="../config",
)
def _hydra_main(cfg: OmegaConf) -> None:
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


def main(argv: list[str] | None = None) -> None:
    # Keep logs streaming in long-running training jobs.
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

    _register_resolvers()
    _enable_tf32()

    parser = argparse.ArgumentParser(
        description="Train models with Hydra config overrides.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Hydra config name to train with, e.g. train_visual_focus_seeker.",
    )
    args, hydra_args = parser.parse_known_args(argv)

    sys.argv = ["seeker train"]
    if args.config_name:
        sys.argv.append(f"--config-name={args.config_name}")
    sys.argv.extend(hydra_args)

    _hydra_main()
