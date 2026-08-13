"""Typed config loading and pretty-print helpers for Seeker model settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf
from seeker.util.path_resolver import load_path_config


_SEEKER_CFG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "seeker_default_params.yaml"
)


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    ckpt_path: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("backbone.name must be set")
        if not self.ckpt_path:
            raise ValueError("backbone.ckpt_path must be set")


@dataclass(frozen=True)
class QueryComposerBaseConfig:
    use_rotation: bool
    task_emb_dim: int
    hidden_mult: int
    proprio_dim: int
    disable_proprio: bool

    def __post_init__(self) -> None:
        if self.task_emb_dim <= 0:
            raise ValueError("query_composer.task_emb_dim must be > 0")
        if self.hidden_mult <= 0:
            raise ValueError("query_composer.hidden_mult must be > 0")
        if self.proprio_dim <= 0:
            raise ValueError("query_composer.proprio_dim must be > 0")


@dataclass(frozen=True)
class QueryComposerConfig(QueryComposerBaseConfig):
    emb_dim: int
    num_robots: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.emb_dim <= 0:
            raise ValueError("query_composer.emb_dim must be > 0")
        if self.num_robots <= 0:
            raise ValueError("query_composer.num_robots must be > 0")


@dataclass(frozen=True)
class IntentRefinerBaseConfig:
    num_refinement_iters: int
    entmax_alpha: float
    hidden_multiplier: int
    disable_head_gating: bool

    def __post_init__(self) -> None:
        if self.num_refinement_iters < 1:
            raise ValueError("intent_refiner.num_refinement_iters must be >= 1")
        if self.entmax_alpha <= 1.0:
            raise ValueError("intent_refiner.entmax_alpha must be > 1.0")
        if self.hidden_multiplier <= 0:
            raise ValueError("intent_refiner.hidden_multiplier must be > 0")


@dataclass(frozen=True)
class IntentRefinerConfig(IntentRefinerBaseConfig):
    emb_dim: int
    num_heads: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.emb_dim <= 0:
            raise ValueError("intent_refiner.emb_dim must be > 0")
        if self.num_heads <= 0:
            raise ValueError("intent_refiner.num_heads must be > 0")


@dataclass(frozen=True)
class SeekerRuntimeConfig:
    top_p: float
    select_n_heads: int

    def __post_init__(self) -> None:
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("seeker.top_p must be in (0, 1]")
        if self.select_n_heads < 1:
            raise ValueError("seeker.select_n_heads must be >= 1")


@dataclass(frozen=True)
class SeekerModelConfig:
    backbone: BackboneConfig
    query_composer: QueryComposerConfig
    intent_refiner: IntentRefinerConfig
    seeker: SeekerRuntimeConfig


@dataclass(frozen=True)
class SeekerBaseConfig:
    backbone: BackboneConfig
    query_composer: QueryComposerBaseConfig
    intent_refiner: IntentRefinerBaseConfig
    seeker: SeekerRuntimeConfig


def _require_mapping(cfg: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = cfg.get(section)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing/invalid section: {section}")
    return value


def _require_keys(section_cfg: Mapping[str, Any], keys: Iterable[str], prefix: str) -> None:
    missing = [f"{prefix}.{k}" for k in keys if k not in section_cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


def _parse_backbone(cfg: Mapping[str, Any]) -> BackboneConfig:
    _require_keys(cfg, ("name", "ckpt_path"), "backbone")
    ckpt = Path(str(cfg["ckpt_path"])).expanduser()
    if not ckpt.is_absolute():
        ckpt = load_path_config().package_root / ckpt
    return BackboneConfig(
        name=str(cfg["name"]),
        ckpt_path=str(ckpt.resolve()),
    )


def _parse_query_composer(cfg: Mapping[str, Any]) -> QueryComposerBaseConfig:
    _require_keys(
        cfg,
        ("use_rotation", "task_emb_dim", "hidden_mult", "proprio_dim", "disable_proprio"),
        "query_composer",
    )
    return QueryComposerBaseConfig(
        use_rotation=bool(cfg["use_rotation"]),
        task_emb_dim=int(cfg["task_emb_dim"]),
        hidden_mult=int(cfg["hidden_mult"]),
        proprio_dim=int(cfg["proprio_dim"]),
        disable_proprio=bool(cfg["disable_proprio"]),
    )


def _parse_intent_refiner(cfg: Mapping[str, Any]) -> IntentRefinerBaseConfig:
    _require_keys(
        cfg,
        ("num_refinement_iters", "entmax_alpha", "hidden_multiplier", "disable_head_gating"),
        "intent_refiner",
    )
    return IntentRefinerBaseConfig(
        num_refinement_iters=int(cfg["num_refinement_iters"]),
        entmax_alpha=float(cfg["entmax_alpha"]),
        hidden_multiplier=int(cfg["hidden_multiplier"]),
        disable_head_gating=bool(cfg["disable_head_gating"]),
    )


def _parse_seeker(cfg: Mapping[str, Any]) -> SeekerRuntimeConfig:
    _require_keys(cfg, ("top_p", "select_n_heads"), "seeker")
    return SeekerRuntimeConfig(
        top_p=float(cfg["top_p"]),
        select_n_heads=int(cfg["select_n_heads"]),
    )


def load_seeker_base_config(
    overrides: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> SeekerBaseConfig:
    cfg_path = config_path or _SEEKER_CFG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Seeker config not found: {cfg_path}")

    cfg_omega = OmegaConf.load(cfg_path)
    if overrides:
        # Resolve interpolations against `overrides`' own config tree before
        # merging: once merged into `cfg_omega`, any interpolation referring
        # to a key outside `seeker_default_params.yaml` would fail to resolve.
        if isinstance(overrides, (DictConfig, ListConfig)):
            overrides = OmegaConf.to_container(overrides, resolve=True)
        cfg_omega = OmegaConf.merge(cfg_omega, OmegaConf.create(dict(overrides)))
    cfg = OmegaConf.to_container(cfg_omega, resolve=True)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Invalid seeker config format in {cfg_path}")

    return SeekerBaseConfig(
        backbone=_parse_backbone(_require_mapping(cfg, "backbone")),
        query_composer=_parse_query_composer(_require_mapping(cfg, "query_composer")),
        intent_refiner=_parse_intent_refiner(_require_mapping(cfg, "intent_refiner")),
        seeker=_parse_seeker(_require_mapping(cfg, "seeker")),
    )


def resolve_seeker_model_config(
    base_cfg: SeekerBaseConfig,
    *,
    emb_dim: int,
    num_heads: int,
    num_robots: int,
) -> SeekerModelConfig:
    """Resolve runtime-derived dimensions into the final Seeker component configs."""
    emb_dim = int(emb_dim)
    num_heads = int(num_heads)
    num_robots = int(num_robots)

    return SeekerModelConfig(
        backbone=base_cfg.backbone,
        query_composer=QueryComposerConfig(
            use_rotation=base_cfg.query_composer.use_rotation,
            task_emb_dim=base_cfg.query_composer.task_emb_dim,
            hidden_mult=base_cfg.query_composer.hidden_mult,
            proprio_dim=base_cfg.query_composer.proprio_dim,
            disable_proprio=base_cfg.query_composer.disable_proprio,
            emb_dim=emb_dim,
            num_robots=num_robots,
        ),
        intent_refiner=IntentRefinerConfig(
            num_refinement_iters=base_cfg.intent_refiner.num_refinement_iters,
            entmax_alpha=base_cfg.intent_refiner.entmax_alpha,
            hidden_multiplier=base_cfg.intent_refiner.hidden_multiplier,
            disable_head_gating=base_cfg.intent_refiner.disable_head_gating,
            emb_dim=emb_dim,
            num_heads=num_heads,
        ),
        seeker=base_cfg.seeker,
    )


def load_seeker_model_config(
    overrides: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
    *,
    emb_dim: int,
    num_heads: int,
    num_robots: int,
) -> SeekerModelConfig:
    """Load YAML config and resolve runtime-derived component dimensions."""
    base_cfg = load_seeker_base_config(overrides=overrides, config_path=config_path)
    return resolve_seeker_model_config(
        base_cfg,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_robots=num_robots,
    )


def build_seeker_pretty_config(
    *,
    view_names: Iterable[str],
    ckpt_path: Optional[str],
    cfg: SeekerModelConfig,
    out_dim: int,
) -> dict[str, Any]:
    def pretty_view_name(name: str) -> str:
        if name == "agentview":
            return "Agent View"
        if name == "eye_in_hand":
            return "Eye-in-Hand"
        return name.replace("_", " ").title()

    stages_str = "coarse -> fine"
    names = list(view_names)
    branch_lines = []
    for i, view in enumerate(names):
        prefix = "└─" if i == len(names) - 1 else "├─"
        branch_lines.append(f"{prefix} {pretty_view_name(view):<12s}: {stages_str}")

    return {
        "Initialization": {
            "Source": "Checkpoint" if ckpt_path else "Random Initialization",
            "Checkpoint Path": ckpt_path,
        },
        "Seeker Tower": {
            "Backbone": f"{cfg.backbone.name} (frozen)",
            "Branches": branch_lines,
        },
        "Query Composer": {
            "Use rotation": cfg.query_composer.use_rotation,
            "Task emb dim": cfg.query_composer.task_emb_dim,
            "Disable proprio": cfg.query_composer.disable_proprio,
        },
        "Intent Refiner": {
            "Refinement iters": cfg.intent_refiner.num_refinement_iters,
            "Entmax alpha": cfg.intent_refiner.entmax_alpha,
            "Head gating": (
                "disabled (uniform)"
                if cfg.intent_refiner.disable_head_gating
                else "intent_alignment (fixed)"
            ),
            "Output dim": out_dim,
        },
        "Masking": {
            "Top-p": cfg.seeker.top_p,
            "Select heads": cfg.seeker.select_n_heads,
        },
    }
