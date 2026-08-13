"""Packaging configuration for the Seeker Python package."""

from pathlib import Path

from setuptools import find_namespace_packages, setup


BASE_DIR = Path(__file__).resolve().parent
README_PATH = BASE_DIR / "README.md"


INSTALL_REQUIRES = [
    "einops>=0.4.1",
    "entmax>=1.3",
    "h5py>=3.8.0",
    "kornia>=0.7.0",
    "lmdb>=1.4.1",
    "numpy>=1.21",
    "omegaconf>=2.2.0",
    "scipy>=1.8.0",
    "torch>=1.9",
    "torchvision>=0.10.0",
    "tqdm>=4.64.0",
]


def read_readme() -> str:
    if README_PATH.is_file():
        return README_PATH.read_text(encoding="utf-8")
    return ""

setup(
    name="seeker",
    version="0.1.0",
    description="Attention from Action for Action: emergent visual bottlenecks in policy learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Zheyu Zhuang et al.",
    license="MIT",
    python_requires=">=3.8",
    packages=find_namespace_packages(
        include=["seeker*"],
        exclude=("tests", "tests.*", "docs", "examples"),
    ),
    include_package_data=True,
    package_data={
        "seeker": [
            "config/*.yaml",
            "config/task/*.yaml",
            "config/method/*.yaml",
            "config/experiment/*.yaml",
        ],
    },
    install_requires=INSTALL_REQUIRES,
    entry_points={
        "console_scripts": [
            "seeker=seeker.scripts.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
