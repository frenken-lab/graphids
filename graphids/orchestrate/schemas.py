"""Pydantic schemas for Monarch pipeline configs.

Validates CLI inputs against ``axes.json`` / ``topology.json`` so typos
crash at config construction, not deep in jsonnet rendering.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal  # noqa: F401 (resolved by model_rebuild)

from pydantic import (  # noqa: F401 (AfterValidator resolved by model_rebuild)
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
)

from graphids.config.constants import (  # noqa: F401 (resolved by model_rebuild)
    PIPELINE_DEFAULTS,
    VALID_FUSION_METHODS,
    VALID_SCALES,
)
from graphids.config.topology import TOPOLOGY  # noqa: F401 (resolved by model_rebuild)
from graphids.orchestrate.planning.recipes import (  # noqa: F401 (resolved by model_rebuild)
    TrainingRunConfig,
    check_in,
)


def check_all_in(valid, label):  # noqa: F401 (resolved by model_rebuild)
    def _v(v):
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(f"Unknown {label}(s): {bad}. Valid: {sorted(valid)}")
        return v

    return _v


_D = PIPELINE_DEFAULTS


class PipelineConfig(BaseModel):
    """What to run in a single Monarch allocation (monarch-run CLI)."""

    model_config = ConfigDict(frozen=True)

    dataset: str = _D.get("dataset", "hcrl_ch")
    seed: int = _D.get("seed", 42)
    scale: Annotated[str, AfterValidator(check_in(VALID_SCALES, "scale"))] = _D.get(
        "scale", "small"
    )
    lake_root: str = ""
    fusion_method: Annotated[
        str, AfterValidator(check_in(VALID_FUSION_METHODS, "fusion_method"))
    ] = _D.get("fusion_method", "bandit")
    stages: Annotated[list[str], AfterValidator(check_all_in(TOPOLOGY.stages, "stage"))] = Field(
        default_factory=lambda: list(_D.get("stages", ["autoencoder", "supervised", "fusion"])),
    )
    conv_type: Literal["gatv2", "gat", "gps"] = _D.get("conv_type", "gatv2")
    variational: bool = _D.get("variational", True)
    loss_fn: Literal["focal", "ce", "weighted_ce"] = _D.get("loss_fn", "focal")
    tla_overrides: dict[str, Any] = Field(default_factory=dict)
    max_retries: int = 2

    def to_training_run(self) -> TrainingRunConfig:
        """Convert CLI fields to a planner-ready TrainingRunConfig."""
        return TrainingRunConfig(
            stages=tuple(self.stages),
            scale=self.scale,
            conv_type=self.conv_type,
            variational=self.variational,
            loss_fn=self.loss_fn,
            fusion_method=self.fusion_method,
        )


# Resolve deferred Annotated annotations (from __future__ import annotations).
PipelineConfig.model_rebuild()


class SweepConfig(BaseModel):
    """Full sweep run configuration (monarch-sweep CLI)."""

    model_config = ConfigDict(frozen=True)

    recipe_path: str
    datasets: list[str] = Field(default_factory=lambda: [_D.get("dataset", "hcrl_ch")])
    seeds: list[int] = Field(default_factory=lambda: [_D.get("seed", 42)])
    lake_root: str = ""
    max_retries: int = 2
    max_concurrent: int = 0  # 0 = all parallel
