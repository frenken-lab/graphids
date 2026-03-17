"""Compose config from YAML layers: defaults -> model_def -> auxiliaries -> CLI overrides."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from .schema import PipelineConfig

CONFIG_DIR = Path(__file__).parent
log = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def resolve(
    model_type: str,
    scale: str,
    auxiliaries: str = "none",
    **cli_overrides,
) -> PipelineConfig:
    # 1. Start with empty dict (Pydantic defaults are the baseline)
    merged: dict = {}

    # 2. Model definition (architecture + scale-specific overrides)
    model_path = CONFIG_DIR / "models" / model_type / f"{scale}.yaml"
    if model_path.exists():
        _deep_merge(merged, yaml.safe_load(model_path.read_text()))

    # 3. Auxiliaries
    if auxiliaries != "none":
        aux_path = CONFIG_DIR / "auxiliaries" / f"{auxiliaries}.yaml"
        if aux_path.exists():
            _deep_merge(merged, yaml.safe_load(aux_path.read_text()))

    # 4. CLI overrides (nested dict from caller)
    if cli_overrides:
        _deep_merge(merged, cli_overrides)

    # 5. Set identity fields
    merged["model_type"] = model_type
    merged["scale"] = scale

    # 5b. Env var overrides for storage paths (shared project storage)
    exp_root = os.environ.get("KD_GAT_EXPERIMENT_ROOT")
    if exp_root:
        merged["experiment_root"] = exp_root

    # 6. Pydantic validates + freezes
    return PipelineConfig.model_validate(merged)


def list_models() -> dict[str, list[str]]:
    """Discover available model types and scales from filesystem.

    Only includes directories whose name matches a valid model_type (vgae, gat, dqn).
    The ``fusion/`` directory contains method-variant configs (dqn.yaml, mlp.yaml, ...),
    not model-type configs, so it is excluded.
    """
    valid_model_types = {"vgae", "gat", "dqn"}
    models = {}
    models_dir = CONFIG_DIR / "models"
    if models_dir.exists():
        for model_dir in sorted(models_dir.iterdir()):
            if model_dir.is_dir() and model_dir.name in valid_model_types:
                scales = [f.stem for f in sorted(model_dir.glob("*.yaml"))]
                if scales:
                    models[model_dir.name] = scales
    return models


def list_auxiliaries() -> list[str]:
    """Discover available auxiliary configs from filesystem."""
    aux_dir = CONFIG_DIR / "auxiliaries"
    if aux_dir.exists():
        return [f.stem for f in sorted(aux_dir.glob("*.yaml"))]
    return ["none"]
