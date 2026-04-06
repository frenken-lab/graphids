"""Shared orchestration data types.

``StageConfig`` is the planner's per-asset value — the input to
``ConfigResolver.resolve`` and the shape every asset materialization
passes to SLURM submission. Pure data, no dagster dependency, no
torch/Lightning imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageConfig:
    """Training config for one asset. Pure data, no dagster dependency."""

    asset_name: str
    stage: str
    model_type: str
    scale: str
    jsonnet_path: str = ""
    model_init_overrides: dict[str, Any] = field(default_factory=dict)
    identity: str = ""
    kd_tag: str = ""
    resource_model: str = ""  # model key for resource lookup (fusion method for fusion stages)
    kd_overrides: dict[str, Any] = field(default_factory=dict)  # raw KDEntry payload
    trainer_overrides: dict[str, str] = field(default_factory=dict)
    stage_overrides: dict[str, str] = field(default_factory=dict)
    resource_overrides: dict[str, str | int] = field(default_factory=dict)
    upstream_asset_names: tuple[str, ...] = ()
    upstream_model_families: dict[str, str] = field(default_factory=dict)
