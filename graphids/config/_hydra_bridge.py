"""Hydra Compose API bridge — config composition via resolve().

Architecture: schema-merge approach.
1. Compose Hydra config with ONLY config group selections (model=X, dataset=Y)
2. Build full-field schema from PipelineConfig Pydantic defaults
3. Merge: schema (base) + Hydra config (YAML overrides)
4. Apply nested overrides via OmegaConf.update(force_add=False) for typo detection
5. Convert to dict → PipelineConfig.model_validate() → frozen

No ++, no _flatten_dict, no _to_hydra_value.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from .schema import PipelineConfig


CONF_DIR = str((Path(__file__).parent / "conf").resolve())

# Keys in Hydra config but not in PipelineConfig — stripped before validation
_HYDRA_ONLY_KEYS = frozenset({"stage"})


# ---------------------------------------------------------------------------
# Hydra Compose API (private)
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
# Schema + merge
# ---------------------------------------------------------------------------


def _build_schema() -> DictConfig:
    """Full-field DictConfig from PipelineConfig defaults.

    Every Pydantic field is present with its default value. This is the base
    that makes OmegaConf.update(force_add=False) work for typo detection.
    """
    from .schema import PipelineConfig

    return OmegaConf.create(PipelineConfig().model_dump())



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
    """Compose config via Hydra → schema merge → Pydantic validation.

    Signature unchanged — zero downstream breakage.
    """
    from .schema import PipelineConfig

    # Config group overrides for Hydra
    group_overrides: list[str] = [
        f"model={model_type}_{scale}",
        f"auxiliary={auxiliaries}",
    ]

    # Known datasets use config group selection; unknown set after merge
    _unknown_dataset: str | None = None
    if dataset is not None:
        ds_yaml = Path(__file__).parent / "conf" / "dataset" / f"{dataset}.yaml"
        if ds_yaml.exists():
            group_overrides.append(f"dataset={dataset}")
        else:
            _unknown_dataset = dataset

    # 1. Hydra compose (config groups only)
    with _hydra_compose(group_overrides) as hydra_cfg:
        pass

    # 2. Schema (all fields with defaults)
    schema = _build_schema()

    # 3. Merge: schema base + Hydra YAML on top
    OmegaConf.set_struct(schema, False)
    merged = OmegaConf.merge(schema, hydra_cfg)

    # 4. Apply programmatic overrides (from kwargs)
    if seed is not None:
        OmegaConf.update(merged, "seed", seed)
    for key, value in config_overrides.items():
        OmegaConf.update(merged, key, value, merge=True, force_add=False)

    # 5. Convert to dict, strip Hydra-only keys, validate
    raw: dict = OmegaConf.to_object(merged)
    for key in _HYDRA_ONLY_KEYS:
        raw.pop(key, None)
    if _unknown_dataset is not None:
        raw["dataset"] = _unknown_dataset

    return PipelineConfig.model_validate(raw)
