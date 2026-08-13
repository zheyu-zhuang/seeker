"""Combined setup command: suite dependencies + release assets."""

from __future__ import annotations

import argparse
import subprocess
import sys

from seeker import REPO_ROOT

from . import setup_assets

SUITE_DEPS_SCRIPT = REPO_ROOT / "seeker" / "scripts" / "setup_suite_deps.sh"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install the pinned robosuite/robomimic/mimicgen suite, then "
            "download release assets (weights/backgrounds/textures). "
            "Equivalent to running setup_suite_deps.sh followed by "
            "'seeker setup-assets'."
        ),
    )
    parser.add_argument(
        "--skip-suite-deps",
        action="store_true",
        help="Skip installing the robosuite/robomimic/mimicgen suite.",
    )
    parser.add_argument(
        "--suite-deps-root",
        default=None,
        help=(
            "Checkout directory for the suite deps (passed through to "
            "setup_suite_deps.sh). Defaults to ../seeker-suite-deps."
        ),
    )
    parser.add_argument("--repo", default=setup_assets.DEFAULT_REPO)
    parser.add_argument("--release-tag", default=setup_assets.DEFAULT_RELEASE_TAG)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-task-cache", action="store_true")
    return parser.parse_args(argv)


def run_suite_deps(deps_root: str | None) -> None:
    if not SUITE_DEPS_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing suite deps script: {SUITE_DEPS_SCRIPT}")

    cmd = ["bash", str(SUITE_DEPS_SCRIPT)]
    if deps_root:
        cmd.append(deps_root)
    print(f"[setup] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.skip_suite_deps:
        print("[setup] Skipping suite dependency install (--skip-suite-deps).")
    else:
        try:
            run_suite_deps(args.suite_deps_root)
        except subprocess.CalledProcessError as exc:
            print(
                f"[setup] Suite dependency install failed (exit {exc.returncode}).",
                file=sys.stderr,
            )
            return exc.returncode
        except FileNotFoundError as exc:
            print(f"[setup] {exc}", file=sys.stderr)
            return 1

    asset_argv = ["--repo", args.repo, "--release-tag", args.release_tag]
    if args.force:
        asset_argv.append("--force")
    if args.skip_task_cache:
        asset_argv.append("--skip-task-cache")
    return setup_assets.main(asset_argv)


if __name__ == "__main__":
    raise SystemExit(main())
