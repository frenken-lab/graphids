"""Hydra Compose API bridge — config composition via resolve().

Responsibilities:
1. Hydra Compose API context manager
2. resolve() — single entry point for config composition
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

if TYPE_CHECKING:

    from .schema import PipelineConfig

log = logging.getLogger(__name__)

CONF_DIR = str((Path(__file__).parent / "conf").resolve())


# ---------------------------------------------------------------------------
# Hydra Compose API
# ---------------------------------------------------------------------------


@contextmanager
def _hydra_compose(overrides: list[str]):
    """Compose a Hydra config with the given overrides.

    Clears GlobalHydra each time to avoid singleton leaks between calls
    (especially in tests).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        yield compose(config_name="config", overrides=overrides)


# ---------------------------------------------------------------------------
# Override list builder
# ---------------------------------------------------------------------------


def _to_hydra_value(v: Any) -> str:
    """Convert a Python value to Hydra override grammar."""
    if isinstance(v, (list, tuple)):
        inner = ",".join(str(x) for x in v)
        return f"[{inner}]"
    if isinstance(v, bool):
        return str(v).lower()
    return str(v)


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten nested dict into Hydra dot-path overrides."""
    items: list[str] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, key))
        else:
            items.append(f"++{key}={_to_hydra_value(v)}")
    return items


# ---------------------------------------------------------------------------
# resolve() — the public API
# ---------------------------------------------------------------------------


def resolve(
    model_type: str = "vgae",
    scale: str = "large",
    auxiliaries: str = "none",
    *,
    dataset: str | None = None,
    seed: int | None = None,
    **config_overrides: Any,
) -> PipelineConfig:
    """Compose config via Hydra → Pydantic validation.

    Args:
        model_type: Architecture type (vgae, gat, dqn).
        scale: Model capacity (large, small).
        auxiliaries: Loss modifier name or "none".
        dataset: Dataset name override.
        seed: Random seed override.
        **config_overrides: Nested dicts or dot-path overrides passed to Hydra.
    """
    from .schema import PipelineConfig

    # Build Hydra override list
    override_list = [
        f"model={model_type}_{scale}",
        f"auxiliary={auxiliaries}",
    ]
    # For known datasets, use config group selection. Unknown datasets
    # (e.g. test fixtures) are handled after composition via OmegaConf.
    _unknown_dataset = None
    if dataset is not None:
        ds_yaml = Path(__file__).parent / "conf" / "dataset" / f"{dataset}.yaml"
        if ds_yaml.exists():
            override_list.append(f"dataset={dataset}")
        else:
            _unknown_dataset = dataset
    if seed is not None:
        override_list.append(f"seed={seed}")

    # Flatten nested dicts (preserves E2E_OVERRIDES, SMOKE_OVERRIDES patterns)
    for key, value in config_overrides.items():
        if isinstance(value, dict):
            override_list.extend(_flatten_dict(value, key))
        else:
            override_list.append(f"++{key}={value}")

    with _hydra_compose(override_list) as hydra_cfg:
        raw: dict = OmegaConf.to_object(hydra_cfg)

    # Remove Hydra-only keys that aren't PipelineConfig fields
    raw.pop("stage", None)

    # For unknown datasets (no config group file), set after composition
    if _unknown_dataset is not None:
        raw["dataset"] = _unknown_dataset

    return PipelineConfig.model_validate(raw)
