"""Pydantic schemas for Monarch pipeline configs.

Validates CLI inputs against ``axes.json`` / ``topology.json`` so typos
crash at config construction, not deep in jsonnet rendering.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphids.config.constants import PIPELINE_DEFAULTS

_D = PIPELINE_DEFAULTS


class PipelineConfig(BaseModel):
    """What to run in a single Monarch allocation (monarch-run CLI)."""

    model_config = ConfigDict(frozen=True)

    dataset: str = _D.get("dataset", "hcrl_ch")
    seed: int = _D.get("seed", 42)
    scale: str = _D.get("scale", "small")
    lake_root: str = ""
    fusion_method: str = _D.get("fusion_method", "bandit")
    stages: list[str] = Field(
        default_factory=lambda: list(_D.get("stages", ["autoencoder", "supervised", "fusion"])),
    )
    conv_type: str = _D.get("conv_type", "gatv2")
    variational: bool = _D.get("variational", True)
    loss_fn: str = _D.get("loss_fn", "focal")
    tla_overrides: dict[str, Any] = Field(default_factory=dict)
    max_retries: int = 2

    @model_validator(mode="after")
    def _validate_axes(self) -> PipelineConfig:
        from graphids.config.constants import VALID_FUSION_METHODS, VALID_SCALES
        from graphids.config.topology import STAGES

        if self.scale not in VALID_SCALES:
            raise ValueError(f"scale={self.scale!r} not in {sorted(VALID_SCALES)}")
        for s in self.stages:
            if s not in STAGES:
                raise ValueError(f"stage={s!r} not in {sorted(STAGES)}")
        if "fusion" in self.stages and self.fusion_method not in VALID_FUSION_METHODS:
            raise ValueError(
                f"fusion_method={self.fusion_method!r} not in {sorted(VALID_FUSION_METHODS)}"
            )
        return self


class SweepConfig(BaseModel):
    """Full sweep run configuration (monarch-sweep CLI)."""

    model_config = ConfigDict(frozen=True)

    recipe_path: str
    datasets: list[str] = Field(default_factory=lambda: [_D.get("dataset", "hcrl_ch")])
    seeds: list[int] = Field(default_factory=lambda: [_D.get("seed", 42)])
    lake_root: str = ""
    max_retries: int = 2
    max_concurrent: int = 0  # 0 = all parallel
