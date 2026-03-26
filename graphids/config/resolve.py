"""Config resolution: compose dataclass defaults + preset YAML + env vars + CLI overrides."""

from __future__ import annotations

import os

import yaml

from jsonargparse import ArgumentParser, Namespace

from .constants import (
    DEFAULT_DATASET,
    DEFAULT_MODEL_TYPE,
    DEFAULT_SCALE,
    DEFAULT_STAGE,
    DEFAULTS_DIR,
    compute_identity_hash,
)
from .defaults.schema import Config


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
