"""Seeker package."""

import logging
from pathlib import Path

logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
WEIGHTS_DIR = REPO_ROOT / ".weights"
DATASETS_DIR = REPO_ROOT / "datasets"
MIMICGEN_DATASETS_DIR = DATASETS_DIR / "mimicgen"
BACKGROUNDS_DIR = DATASETS_DIR / "backgrounds"
TEXTURES_DIR = DATASETS_DIR / "textures"
