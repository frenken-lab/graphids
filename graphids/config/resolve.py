"""Config resolution: compose dataclass defaults + preset YAML + env vars + CLI overrides."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

from jsonargparse import ArgumentParser, Namespace

from .constants import (
    CONFIG_DIR,
    DEFAULT_DATASET,
    DEFAULT_MODEL_TYPE,
    DEFAULT_SCALE,
    DEFAULT_STAGE,
    DEFAULTS_DIR,
    PIPELINE_YAML,
)
from .defaults.schema import Config


def to_namespace(cfg):
    """Convert dict (from checkpoint reload) to jsonargparse Namespace.

    Recursively converts nested dicts and list items. No-op on Namespace.
    """
    if isinstance(cfg, Namespace):
        return cfg
    if isinstance(cfg, dict):
        return Namespace(**{k: to_namespace(v) for k, v in cfg.items()})
    if isinstance(cfg, list):
        return [to_namespace(v) for v in cfg]
    return cfg


def compute_identity_hash(stage: str, cfg) -> str:
    """Compute identity hash for a stage from its identity_keys.

    Returns ``"_<8-char-hex>"`` or ``""`` if the stage has no identity keys.
    """
    stage_def = PIPELINE_YAML.get("stages", {}).get(stage, {})
    keys = stage_def.get("identity_keys", [])
    if not keys:
        return ""

    def _get(dotted_key, default=None):
        cur = cfg
        for part in dotted_key.split("."):
            if cur is None:
                return default
            cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        return cur if cur is not None else default

    unresolved = [k for k in keys if _get(k) is None]
    if unresolved:
        import structlog
        structlog.get_logger().warning("identity_key_unresolved", stage=stage, keys=unresolved)
    pairs = [f"{k}={_get(k, '_default_')}" for k in sorted(keys)]
    return "_" + hashlib.sha256("|".join(pairs).encode()).hexdigest()[:8]


def data_dir(lake_root: str, dataset: str) -> Path:
    """Raw data directory. Tries lake, falls back to local."""
    from .constants import PREPROCESSING_VERSION
    candidate = Path(lake_root) / "raw" / dataset
    if candidate.exists():
        return candidate
    return Path("data") / "automotive" / dataset


def cache_dir(lake_root: str, dataset: str) -> Path:
    """Processed-graph cache directory."""
    from .constants import PREPROCESSING_VERSION
    return Path(lake_root) / "cache" / f"v{PREPROCESSING_VERSION}" / dataset


def _compute_derived(cfg: Namespace) -> None:
    """Fill path fields after all overrides are applied."""
    user = os.environ.get("USER", "unknown")
    cfg._tier = f"dev/{user}"
    cfg._output_base = f"{cfg.lake_root}/{cfg._tier}/{cfg.dataset}"

    _CKPT_STAGES = {
        "vgae": "autoencoder",
        "gat": cfg.gat_stage,
        "dqn": "fusion",
        "dgi": "autoencoder",
        "temporal": "temporal",
    }
    _CKPT_MODEL = {"temporal": "gat"}
    cfg.checkpoints = Namespace(**{
        model: (
            f"{cfg._output_base}/{_CKPT_MODEL.get(model, model)}_{cfg.scale}_{stage}"
            f"{compute_identity_hash(stage, cfg)}/seed_{cfg.seed}/best_model.ckpt"
        )
        for model, stage in _CKPT_STAGES.items()
    })


def resolve(*overrides: str) -> Namespace:
    """Compose config: dataclass defaults + YAML preset + CLI overrides.

    jsonargparse handles type coercion, nested dataclass flattening, and
    YAML/CLI merge. Preset file selected from model_type + scale.
    """
    # Extract top-level overrides for preset lookup
    top = {}
    for o in overrides:
        if "=" in o and "." not in o:
            k, v = o.split("=", 1)
            top[k] = v

    model_type = top.get("model_type", DEFAULT_MODEL_TYPE)
    scale = top.get("scale", DEFAULT_SCALE)
    preset_key = f"{model_type}_{scale}"

    # Load preset from single presets.yaml, write to temp file for jsonargparse
    import tempfile
    presets_path = DEFAULTS_DIR / "presets.yaml"
    presets = yaml.safe_load(presets_path.read_text()) if presets_path.exists() else {}
    preset = presets.get(preset_key, {})
    config_files = []
    if preset:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(preset, tmp, default_flow_style=False)
        tmp.close()
        config_files = [tmp.name]

    parser = ArgumentParser(default_config_files=config_files, env_prefix="KD_GAT", default_env=True)
    parser.add_class_arguments(Config, nested_key=None)

    # Inject pipeline-derived defaults (pipeline.yaml is the source of truth)
    parser.set_defaults({
        "model_type": DEFAULT_MODEL_TYPE,
        "scale": DEFAULT_SCALE,
        "stage": DEFAULT_STAGE,
        "dataset": DEFAULT_DATASET,
    })

    args = [f"--{o}" if "=" in o and not o.startswith("-") else o for o in overrides]
    cfg = parser.parse_args(args)

    _compute_derived(cfg)
    return cfg
