"""Shared orchestration data types.

``StageConfig`` is the planner's per-asset value — the input to
``ConfigResolver.resolve`` and the shape every asset materialization
passes to SLURM submission. Pure data, no dagster dependency, no
torch/Lightning imports.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StageConfig(BaseModel):
    """Training config for one asset. Pure data, no dagster dependency."""

    model_config = ConfigDict(frozen=True)

    asset_name: str
    stage: str
    model_type: str
    scale: str
    jsonnet_path: str = ""
    model_init_overrides: dict[str, Any] = Field(default_factory=dict)
    identity: str = ""
    kd_tag: str = ""
    resource_model: str = ""  # model key for resource lookup (fusion method for fusion stages)
    kd_overrides: dict[str, Any] = Field(default_factory=dict)  # raw KDEntry payload
    trainer_overrides: dict[str, Any] = Field(default_factory=dict)
    stage_overrides: dict[str, Any] = Field(default_factory=dict)
    resource_overrides: dict[str, str | int] = Field(default_factory=dict)
    upstream_asset_names: tuple[str, ...] = ()
    upstream_model_families: dict[str, str] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (Monarch endpoint args must be serializable)."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageConfig:
        """Reconstruct from a dict (inverse of ``to_dict``)."""
        return cls.model_validate(d)
