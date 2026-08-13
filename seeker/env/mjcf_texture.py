"""MJCF table-texture sampling and XML patching utilities."""

import os
import glob
import hashlib
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

from seeker import TEXTURES_DIR


def _default_texture_dir() -> str:
    return str(TEXTURES_DIR)

def list_texture_files(texture_dir: Optional[str] = None) -> List[str]:
    """List texture image files under `texture_dir`."""
    if texture_dir is None:
        texture_dir = _default_texture_dir()
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(texture_dir, e)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No texture images found in: {texture_dir}")
    return [str(Path(p).resolve()) for p in files]


_SAMPLE_COUNTER: Dict[Tuple[str, bool], int] = {}


def _mujoco_safe_texture_file(texture_file: str) -> str:
    """Rewrite texture to a sanitized PNG path that MuJoCo can load reliably."""
    src = Path(texture_file).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Texture file not found: {src}")

    stat = src.stat()
    key = f"{src}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "seeker_mujoco_textures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{src.stem}_{digest}.png"
    if dst.is_file():
        return str(dst)

    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read texture image: {src}")

    tmp = cache_dir / f".{dst.name}.{os.getpid()}.tmp.png"
    if not cv2.imwrite(str(tmp), image):
        raise ValueError(f"Failed to write sanitized texture image: {tmp}")
    os.replace(tmp, dst)
    return str(dst)


def _get_key(texture_dir: str, eval: bool) -> Tuple[str, bool]:
    return (str(Path(texture_dir).resolve()), bool(eval))


def get_next_texture(
    texture_dir: Optional[str] = None,
    *,
    eval: bool = False,
) -> str:
    """Return textures deterministically in sorted cyclic order."""
    if texture_dir is None:
        texture_dir = _default_texture_dir()
    split_dir = Path(texture_dir) / ("eval" if eval else "train")
    files = list_texture_files(split_dir)

    # Order by filename (not full path) to be robust to directory differences.
    files = sorted(files, key=lambda p: Path(p).name)

    n = len(files)
    if n == 0:
        raise FileNotFoundError(f"No texture files found in: {split_dir}")

    key = _get_key(texture_dir, eval)

    # Initialize counter if missing or if file count changed
    if key not in _SAMPLE_COUNTER or _SAMPLE_COUNTER[key] >= n:
        _SAMPLE_COUNTER[key] = 0

    c = _SAMPLE_COUNTER[key]
    _SAMPLE_COUNTER[key] = c + 1  # advance for next call (wrap handled above)
    return files[c]


def _find_geom_by_name(root: ET.Element, geom_name: str) -> Optional[ET.Element]:
    for geom in root.findall(".//geom"):
        if geom.get("name") == geom_name:
            return geom
    return None


def _ensure_texture_and_material(
    root: ET.Element,
    texture_name: str,
    material_name: str,
    texture_file: str,
    texrepeat: Tuple[int, int] = (1, 1),
) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # texture
    tex = None
    for t in asset.findall("texture"):
        if t.get("name") == texture_name:
            tex = t
            break
    if tex is None:
        tex = ET.SubElement(asset, "texture")

    tex.set("type", "2d")
    tex.set("name", texture_name)
    tex.set("file", _mujoco_safe_texture_file(texture_file))

    # material
    mat = None
    for m in asset.findall("material"):
        if m.get("name") == material_name:
            mat = m
            break
    if mat is None:
        mat = ET.SubElement(asset, "material")

    mat.set("name", material_name)
    mat.set("texture", texture_name)
    mat.set("texuniform", "false")
    mat.set("texrepeat", f"{texrepeat[0]} {texrepeat[1]}")


def apply_table_texture_to_xml(
    model_xml: str,
    *,
    texture_file: str,
    table_geom_name: str = "table_visual",
    texture_name: str = "table_tex_dyn",
    material_name: str = "table_mat_dyn",
    texrepeat: Tuple[int, int] = (1, 1),
    also_apply_to_collision: bool = False,
    also_apply_to_legs: bool = False,
) -> str:
    """Apply `texture_file` to target table geom in an MJCF XML string."""
    root = ET.fromstring(model_xml)

    _ensure_texture_and_material(
        root,
        texture_name=texture_name,
        material_name=material_name,
        texture_file=texture_file,
        texrepeat=texrepeat,
    )

    g = _find_geom_by_name(root, table_geom_name)
    if g is None:
        raise KeyError(f"Could not find geom named '{table_geom_name}' in MJCF")
    g.set("material", material_name)

    if also_apply_to_collision:
        g_col = _find_geom_by_name(root, "table_collision")
        if g_col is not None:
            g_col.set("material", material_name)

    if also_apply_to_legs:
        for leg in (
            "table_leg1_visual",
            "table_leg2_visual",
            "table_leg3_visual",
            "table_leg4_visual",
        ):
            lg = _find_geom_by_name(root, leg)
            if lg is not None:
                lg.set("material", material_name)

    return ET.tostring(root, encoding="unicode")


def apply_table_texture(
    model_file_or_xml: str,
    *,
    texture_file: str,
    table_geom_name: str = "table_visual",
    texture_name: str = "table_tex_dyn",
    material_name: str = "table_mat_dyn",
    texrepeat: Tuple[int, int] = (1, 1),
    also_apply_to_collision: bool = False,
    also_apply_to_legs: bool = False,
) -> str:
    """Patch MJCF from xml-string or xml-path and return patched XML text."""
    is_xml_string = str(model_file_or_xml).lstrip().startswith("<")
    if is_xml_string:
        xml_in = model_file_or_xml
    else:
        p = Path(model_file_or_xml).expanduser()
        xml_in = p.read_text()

    return apply_table_texture_to_xml(
        xml_in,
        texture_file=texture_file,
        table_geom_name=table_geom_name,
        texture_name=texture_name,
        material_name=material_name,
        texrepeat=texrepeat,
        also_apply_to_collision=also_apply_to_collision,
        also_apply_to_legs=also_apply_to_legs,
    )
