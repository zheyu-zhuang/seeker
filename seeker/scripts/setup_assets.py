"""Asset setup command for Seeker CLI."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from seeker import BACKGROUNDS_DIR, REPO_ROOT, TEXTURES_DIR, WEIGHTS_DIR


DEFAULT_REPO = "zheyu-zhuang/seeker"
DEFAULT_RELEASE_TAG = "assets"
SEEKER_WEIGHTS_ASSET = "seeker.mimicgen.pth"
DINO_WEIGHTS_ASSET = "dinov3.vits16plus.pth"
RVT2_HEATMAP_WEIGHTS_ASSET = "rvt2_heatmap.mimicgen.pth"

# See seeker/model/WEIGHTS.md for what these checkpoints are and how they
# were produced.
EXPECTED_SHA256 = {
    SEEKER_WEIGHTS_ASSET: "b0aa9d7272e8b93ddccc402959969eb52ae075e8595d0ddd478a1b39c5aacda1",
    DINO_WEIGHTS_ASSET: "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea",
    RVT2_HEATMAP_WEIGHTS_ASSET: "996ea845bcbb1fc4b5a9e8c66671d83acafcbb89113f74009bf81bc94696212e",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Set up Seeker release assets (backgrounds/textures + checkpoint). "
            "Assets must come from a GitHub release, not just a branch."
        ),
        epilog=(
            "Private release:\n"
            "  gh auth login\n"
            "  seeker setup-assets --repo <owner/private-repo> --release-tag <tag>\n\n"
            "If the assets only exist on a dev branch and are not attached to a "
            "GitHub release, this command will not work. In that case, publish a "
            "release or place the files manually."
        ),
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="GitHub repo in owner/name format.",
    )
    parser.add_argument(
        "--release-tag",
        default=DEFAULT_RELEASE_TAG,
        help="GitHub release tag for Seeker assets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--skip-task-cache",
        action="store_true",
        help="Do not prebuild the CLIP task embedding cache after asset setup.",
    )
    return parser.parse_args(argv)


def top_level_roots(zf: zipfile.ZipFile) -> set[str]:
    roots = set()
    for name in zf.namelist():
        clean = name.strip("/")
        if clean:
            roots.add(clean.split("/", 1)[0])
    return roots


def extract_archive(archive: Path, target_dir: Path) -> None:
    target_name = target_dir.name

    with zipfile.ZipFile(archive, "r") as zf:
        roots = top_level_roots(zf)
        if target_name in roots:
            extract_root = target_dir.parent
        else:
            extract_root = target_dir

        extract_root.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_root)

    target_dir.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        print(f"[setup_assets] Skip existing: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    print(f"[setup_assets] Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    print(f"[setup_assets] Saved: {dst}")


def download_release_asset(
    *,
    repo: str,
    release_tag: str,
    asset_name: str,
    dst: Path,
    force: bool,
) -> None:
    if dst.exists() and not force:
        print(f"[setup_assets] Skip existing: {dst}")
        return

    url = f"https://github.com/{repo}/releases/download/{release_tag}/{asset_name}"
    try:
        download_file(url, dst, force=True)
        return
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError(
            "Direct release download returned 404. This usually means the repo/tag/"
            "asset is wrong, or the release is private. For private releases, "
            "install GitHub CLI and run 'gh auth login', then retry with the "
            "correct --repo and --release-tag. If the asset only exists on a branch, "
            f"publish a release first. Missing asset: '{asset_name}'."
        )

    print(
        "[setup_assets] Direct URL returned 404; trying authenticated "
        f"download via gh for {asset_name}"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="seeker_gh_dl_") as tmp:
        cmd = [gh, "release", "download", release_tag, "-R", repo, "-p", asset_name, "-D", tmp]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip() or "unknown gh error"
            raise RuntimeError(
                "Authenticated GitHub release download failed. Check that you have "
                "access to the private repo, ran 'gh auth login', and provided the "
                f"correct --repo/--release-tag. gh error: {stderr}"
            )
        src = Path(tmp) / asset_name
        if not src.exists():
            raise RuntimeError(f"gh reported success but asset missing in temp dir: {asset_name}")
        shutil.move(str(src), dst)
        print(f"[setup_assets] Saved via gh: {dst}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, asset_name: str) -> None:
    expected = EXPECTED_SHA256.get(asset_name)
    if expected is None:
        return

    actual = sha256_of(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for '{asset_name}': expected {expected}, got "
            f"{actual}. The downloaded file was deleted; re-run setup-assets "
            "to retry, or check --repo/--release-tag if you expect a "
            "different checkpoint."
        )
    print(f"[setup_assets] Checksum OK: {asset_name}")


def warm_task_embedding_cache() -> None:
    """Build the task embedding cache before multi-worker jobs need it."""

    try:
        from seeker.util.task_meta import _default_cache_path, setup_task_embedding_cache

        cache_path = _default_cache_path()
        if cache_path.exists():
            print(f"[setup_assets] Task embedding cache exists: {cache_path}")
            setup_task_embedding_cache()
        else:
            print(f"[setup_assets] Building task embedding cache: {cache_path}")
            setup_task_embedding_cache()
            print(f"[setup_assets] Task embedding cache saved: {cache_path}")
    except Exception as exc:
        print(
            "[setup_assets] WARNING: task embedding cache was not built. "
            "It will be built lazily by rerender/eval with a file lock. "
            f"Reason: {exc}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    release_base = f"https://github.com/{args.repo}/releases/download/{args.release_tag}"

    seeker_weights = WEIGHTS_DIR / SEEKER_WEIGHTS_ASSET
    dinov3_weights = WEIGHTS_DIR / DINO_WEIGHTS_ASSET
    rvt2_heatmap_weights = WEIGHTS_DIR / RVT2_HEATMAP_WEIGHTS_ASSET

    print(f"[setup_assets] Repo root       : {REPO_ROOT}")
    print(f"[setup_assets] Backgrounds dir : {BACKGROUNDS_DIR}")
    print(f"[setup_assets] Textures dir    : {TEXTURES_DIR}")
    print(f"[setup_assets] Weights dir     : {WEIGHTS_DIR}")
    print(f"[setup_assets] Release source  : {release_base}")

    try:
        with tempfile.TemporaryDirectory(prefix="seeker_asset_dl_") as temp_dir:
            temp_root = Path(temp_dir)
            backgrounds_zip = temp_root / "backgrounds.zip"
            textures_zip = temp_root / "textures.zip"

            download_release_asset(
                repo=args.repo,
                release_tag=args.release_tag,
                asset_name="backgrounds.zip",
                dst=backgrounds_zip,
                force=args.force,
            )
            download_release_asset(
                repo=args.repo,
                release_tag=args.release_tag,
                asset_name="textures.zip",
                dst=textures_zip,
                force=args.force,
            )

            print("[setup_assets] Extracting backgrounds archive")
            extract_archive(backgrounds_zip, BACKGROUNDS_DIR)
            print("[setup_assets] Extracting textures archive")
            extract_archive(textures_zip, TEXTURES_DIR)

        download_release_asset(
            repo=args.repo,
            release_tag=args.release_tag,
            asset_name=SEEKER_WEIGHTS_ASSET,
            dst=seeker_weights,
            force=args.force,
        )
        verify_checksum(seeker_weights, SEEKER_WEIGHTS_ASSET)
        download_release_asset(
            repo=args.repo,
            release_tag=args.release_tag,
            asset_name=DINO_WEIGHTS_ASSET,
            dst=dinov3_weights,
            force=args.force,
        )
        verify_checksum(dinov3_weights, DINO_WEIGHTS_ASSET)
        download_release_asset(
            repo=args.repo,
            release_tag=args.release_tag,
            asset_name=RVT2_HEATMAP_WEIGHTS_ASSET,
            dst=rvt2_heatmap_weights,
            force=args.force,
        )
        verify_checksum(rvt2_heatmap_weights, RVT2_HEATMAP_WEIGHTS_ASSET)
        if not args.skip_task_cache:
            warm_task_embedding_cache()
    except Exception as exc:
        print(f"[setup_assets] Failed: {exc}", file=sys.stderr)
        return 1

    print("[setup_assets] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
