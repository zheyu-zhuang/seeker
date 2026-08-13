"""Task metadata and CLIP task-embedding cache helpers."""

import os
import re
import tempfile
import threading
import time
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Sequence, Union

import clip
import numpy as np
import torch

from seeker.util.path_resolver import load_path_config

# Task descriptions (MimicGen)

MIMICGEN_DESCRIPTIONS = {
    "coffee_preparation": "Make coffee using the coffee machine and a pod",
    "mug_cleanup": "Store the mug inside the drawer",
    "square": "Insert the square nut onto the square peg",
    "nut_assembly": "Assemble both square and round nuts onto their pegs",
    "stack_three": "Stack three blocks into a vertical tower",
    "three_piece_assembly": "Assemble the three toy pieces together",
    "pick_place": "Collect all objects and place them into the container",
    "threading": "Thread the needle through the eye",
}

ROBOT_ASSIGMNENT = {
    "coffee_preparation": "franka",
    "mug_cleanup": "franka",
    "square": "franka",
    "nut_assembly": "sawyer",
    "stack_three": "franka",
    "three_piece_assembly": "franka",
    "pick_place": "sawyer",
    "threading": "franka",
}

ROBOT_NAME_TO_ID = {"franka": 0, "sawyer": 1}

NUM_ROBOTS = len(ROBOT_NAME_TO_ID)

# Process-local cache
_TASK_CACHE: Optional[Tuple[Dict[str, int], np.ndarray]] = None
_TASK_TOKEN_CACHE: Optional[Tuple[Dict[str, int], np.ndarray]] = None
_WARNED_NO_CACHE = False
_TASK_CACHE_LOCK = threading.RLock()
_LOCK_POLL_SECONDS = 0.1


def _default_cache_path() -> Path:
    return load_path_config().task_emb_cache_path


def encode_text_clip(task_descriptions: Union[list, str], device="cpu"):
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    if isinstance(task_descriptions, str):
        task_descriptions = [task_descriptions]

    with torch.no_grad():
        text_tokens = clip.tokenize(task_descriptions).to(device)
        text_embedding = clip_model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding.to(torch.float32).to(device)


def encode_text_clip_tokens(task_descriptions: Union[list, str], device="cpu"):
    """Return contextual CLIP token embeddings, shape [B, 77, 512]."""
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    if isinstance(task_descriptions, str):
        task_descriptions = [task_descriptions]

    with torch.no_grad():
        text_tokens = clip.tokenize(task_descriptions).to(device)
        x = clip_model.token_embedding(text_tokens).type(clip_model.dtype)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x).type(clip_model.dtype)
    return x.to(torch.float32).to(device)


def _normalize_env_name(env_name: str) -> str:
    name = env_name.strip()
    name = re.sub(r"(_)?(d\d+|v\d+)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"_+", "_", name)
    return name.lower()


def env_name_to_instruction(env_name: str) -> str:
    norm = _normalize_env_name(env_name)
    for key, desc in MIMICGEN_DESCRIPTIONS.items():
        if key in norm:
            return desc.lower()
    raise ValueError(f"Unknown env name '{env_name}' (normalized '{norm}')")


class _TaskEmbeddingCacheFileLock:
    """Small cross-process lock for the task embedding cache file."""

    def __init__(self, cache_path: Path) -> None:
        self.lock_path = cache_path.with_name(cache_path.name + ".lock")
        self._file = None
        self._fd: Optional[int] = None

    def __enter__(self) -> "_TaskEmbeddingCacheFileLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            import fcntl

            self._file = self.lock_path.open("a+")
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
            return self

        while True:
            try:
                self._fd = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                )
                return self
            except FileExistsError:
                time.sleep(_LOCK_POLL_SECONDS)

    def __exit__(self, exc_type, exc, tb) -> None:
        if os.name == "posix":
            import fcntl

            assert self._file is not None
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None
            return

        assert self._fd is not None
        os.close(self._fd)
        self._fd = None
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass


def _write_task_embedding_cache_npz(
    path: Path,
    texts: Sequence[str],
    embeddings: np.ndarray,
    token_embeddings: np.ndarray,
) -> None:
    """Atomically write the task cache under the caller-held file lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp.npz",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        np.savez(
            str(tmp_path),
            texts=np.array([str(t).lower() for t in texts], dtype=object),
            embeddings=embeddings.astype(np.float32, copy=False),
            token_embeddings=token_embeddings.astype(np.float32, copy=False),
        )
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_task_embedding_cache_npz(
    path: Path,
    *,
    require_token_embeddings: bool,
) -> Optional[Tuple[List[str], np.ndarray, Optional[np.ndarray]]]:
    if not path.exists():
        return None

    try:
        with np.load(str(path), allow_pickle=True) as z:
            texts = [str(t).lower() for t in z["texts"].tolist()]
            embeddings = z["embeddings"].astype(np.float32)
            token_embeddings = None
            if "token_embeddings" in z.files:
                token_embeddings = z["token_embeddings"].astype(np.float32)
            elif require_token_embeddings:
                return None
            return texts, embeddings, token_embeddings
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        warnings.warn(
            f"[TaskEmbeddingCache] Ignoring invalid cache at {path}: {exc}",
            stacklevel=2,
        )
        return None


def _build_task_embedding_arrays() -> Tuple[List[str], np.ndarray, np.ndarray]:
    keys = sorted(MIMICGEN_DESCRIPTIONS.keys())
    texts = [MIMICGEN_DESCRIPTIONS[k].lower() for k in keys]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        emb = encode_text_clip(texts, device=device)
        token_emb = encode_text_clip_tokens(texts, device=device)

    return (
        texts,
        emb.cpu().numpy().astype(np.float32),
        token_emb.cpu().numpy().astype(np.float32),
    )


def setup_task_embedding_cache(cache_path: Optional[str] = None) -> None:
    """
    Single entrypoint:
    - If cache exists: load it.
    - Else: build it (encode all task descriptions), save it, then load it.
    """
    global _TASK_CACHE, _TASK_TOKEN_CACHE

    with _TASK_CACHE_LOCK:
        p = (
            _default_cache_path()
            if cache_path is None
            else Path(cache_path).expanduser().resolve()
        )
        p.parent.mkdir(parents=True, exist_ok=True)

        with _TaskEmbeddingCacheFileLock(p):
            cache = _load_task_embedding_cache_npz(
                p,
                require_token_embeddings=False,
            )
            if cache is None:
                texts, emb, token_emb = _build_task_embedding_arrays()
                _write_task_embedding_cache_npz(p, texts, emb, token_emb)
            else:
                texts, emb, token_emb = cache
                if token_emb is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    with torch.no_grad():
                        token_emb_tensor = encode_text_clip_tokens(texts, device=device)
                    token_emb = token_emb_tensor.cpu().numpy().astype(np.float32)
                    _write_task_embedding_cache_npz(p, texts, emb, token_emb)

            loaded = _load_task_embedding_cache_npz(
                p,
                require_token_embeddings=True,
            )
            if loaded is None:
                raise RuntimeError(f"Failed to load task embedding cache at {p}")

        texts, emb, token_emb_loaded = loaded
        assert token_emb_loaded is not None
        _TASK_CACHE = ({t: i for i, t in enumerate(texts)}, emb)
        _TASK_TOKEN_CACHE = ({t: i for i, t in enumerate(texts)}, token_emb_loaded)


def _text_to_task_embedding(text: Union[str, Sequence[str]]) -> torch.Tensor:
    """Shared implementation for env_name_to_task_embedding / instruction_to_task_embedding.
    Supports:
      - str -> (D,)
      - list[str]/tuple[str]/Sequence[str] -> (B, D)
    """
    global _WARNED_NO_CACHE

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalize to a list of keys
    if isinstance(text, str):
        keys = [text.strip().lower()]
        single = True
    else:
        keys = [t.strip().lower() for t in text]
        single = False

    out: List[torch.Tensor] = [None] * len(keys)  # type: ignore
    missing: List[int] = []

    # Try cache first
    if _TASK_CACHE is not None:
        text2idx, emb = _TASK_CACHE  # emb: np.ndarray [N, D]
        for i, k in enumerate(keys):
            idx = text2idx.get(k)
            if idx is not None:
                out[i] = torch.from_numpy(emb[idx]).to(device=device)
            else:
                missing.append(i)
    else:
        missing = list(range(len(keys)))

    # Runtime fallback for misses
    if missing:
        if not _WARNED_NO_CACHE:
            warnings.warn(
                "[TaskEmbeddingCache] Cache not initialized or missing entry; using runtime CLIP. "
                "Call setup_task_embedding_cache() once in your entrypoint to avoid this.",
                stacklevel=2,
            )
            _WARNED_NO_CACHE = True

        for i in missing:
            out[i] = encode_text_clip(keys[i], device=device)

    # Stack / unwrap
    if single:
        return out[0]
    return torch.stack(out, dim=0)


def _text_to_task_language_tokens(text: Union[str, Sequence[str]]) -> torch.Tensor:
    """Return contextual CLIP language tokens.

    Supports:
      - str -> (77, D)
      - sequence[str] -> (B, 77, D)
    """
    global _WARNED_NO_CACHE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(text, str):
        keys = [text.strip().lower()]
        single = True
    else:
        keys = [t.strip().lower() for t in text]
        single = False

    out: List[torch.Tensor] = [None] * len(keys)  # type: ignore
    missing: List[int] = []
    if _TASK_TOKEN_CACHE is not None:
        text2idx, emb = _TASK_TOKEN_CACHE
        for i, k in enumerate(keys):
            idx = text2idx.get(k)
            if idx is not None:
                out[i] = torch.from_numpy(emb[idx]).to(device=device)
            else:
                missing.append(i)
    else:
        missing = list(range(len(keys)))

    if missing:
        if not _WARNED_NO_CACHE:
            warnings.warn(
                "[TaskEmbeddingCache] Cache not initialized or missing entry; using runtime CLIP. "
                "Call setup_task_embedding_cache() once in your entrypoint to avoid this.",
                stacklevel=2,
            )
            _WARNED_NO_CACHE = True
        for i in missing:
            out[i] = encode_text_clip_tokens(keys[i], device=device)[0]

    if single:
        return out[0]
    return torch.stack(out, dim=0)


def env_name_to_task_embedding(env_name: str) -> torch.Tensor:
    """
    Cached if `setup_task_embedding_cache()` has run, otherwise runtime CLIP.
    """
    instruction = env_name_to_instruction(env_name)
    return _text_to_task_embedding(instruction)


def env_name_to_robot(env_name: str) -> str:
    """
    Get robot assignment from env name.
    """
    norm = _normalize_env_name(env_name)
    for key, robot in ROBOT_ASSIGMNENT.items():
        if key in norm:
            return robot.lower()
    raise ValueError(f"Unknown env name '{env_name}' (normalized '{norm}')")


def env_name_to_robot_id(env_name: str) -> int:
    """
    Get robot index from env name.
    """
    robot = env_name_to_robot(env_name)
    if robot in ROBOT_NAME_TO_ID:
        return ROBOT_NAME_TO_ID[robot]
    else:
        raise ValueError(f"Unknown robot type: {robot}")


def instruction_to_task_embedding(instruction: str) -> torch.Tensor:
    """
    Cached if `setup_task_embedding_cache()` has run, otherwise runtime CLIP.
    """
    return _text_to_task_embedding(instruction)


def instruction_to_task_language_tokens(instruction: str) -> torch.Tensor:
    """Return contextual CLIP token embeddings for one instruction."""
    return _text_to_task_language_tokens(instruction)


def env_name_to_meta(env_name: str) -> Dict[str, Union[str, int, torch.Tensor]]:
    """
    Get all task meta info from env name.
    Returns dict with keys:
      - "instruction": str
      - "robot": str
      - "robot_id": int
      - "task_embedding": torch.Tensor (D,)
      - "task_language_tokens": torch.Tensor (77, D)
    """
    instruction = env_name_to_instruction(env_name)
    robot = env_name_to_robot(env_name)
    robot_id = ROBOT_NAME_TO_ID[robot]
    task_embedding = env_name_to_task_embedding(env_name)
    task_language_tokens = instruction_to_task_language_tokens(instruction)

    return {
        "instruction": instruction,
        "robot": robot,
        "robot_id": robot_id,
        "task_embedding": task_embedding,
        "task_language_tokens": task_language_tokens,
    }
