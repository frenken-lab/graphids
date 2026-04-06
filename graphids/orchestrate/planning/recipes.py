"""Recipe envelope Pydantic + expansion facade.

A recipe is a Jsonnet file under ``configs/recipes/*.jsonnet`` that
declares a set of sweeps and/or a selection block. It renders to a raw
dict (via ``graphids.config.jsonnet.render``) whose shape is validated
here through ``_RecipeEnvelope`` before being handed to the Jsonnet-side
expansion (``configs/recipes/_expand.jsonnet``) that produces
``{defaults, configs, sweep, trainer_overrides, stage_overrides,
resource_overrides}``.

Pydantic survives the YAML→Jsonnet migration because Jsonnet has no
enums, no typed fields, and no ``extra="forbid"``. The envelope schema
catches typos in user-written recipe files that Jsonnet happily accepts.

``TrainingRunConfig`` is the typed view of the ``defaults`` block used
by ``enumerate_assets`` when planning StageConfigs. ``KDEntry`` is the
KD sub-schema referenced inside sweep blocks and defaults.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from graphids.config.constants import (
    CONFIG_DIR,
    VALID_FUSION_METHODS,
    VALID_MODEL_TYPES,
    VALID_SCALES,
)
from graphids.config.jsonnet import render
from graphids.config.topology import STAGES


class KDEntry(BaseModel):
    """Recipe-side KD config schema — superset of ``KDAuxiliary``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["kd"] = "kd"
    alpha: float = 0.7
    teacher_config: str | None = None
    teacher_scale: str = "large"
    temperature: float | None = None
    model_path: str | None = None
    vgae_latent_weight: float | None = None
    vgae_recon_weight: float | None = None

    @field_validator("teacher_scale")
    @classmethod
    def _valid_scale(cls, v: str) -> str:
        if v not in VALID_SCALES:
            raise ValueError(f"teacher_scale={v!r} not in {sorted(VALID_SCALES)}")
        return v

    @field_validator("alpha")
    @classmethod
    def _alpha_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"alpha={v} must be in [0, 1]")
        return v

    @field_validator("temperature")
    @classmethod
    def _temperature_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"temperature={v} must be positive")
        return v


class TrainingRunConfig(BaseModel):
    """Typed boundary input for a training run identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stages: tuple[str, ...] = ("autoencoder", "supervised", "fusion")
    scale: str = "small"
    conv_type: str = "gatv2"
    loss_fn: str = "focal"
    fusion_method: str = "bandit"
    variational: bool = True
    model_type: str | None = None
    auxiliaries: tuple[KDEntry, ...] = ()

    @field_validator("stages", mode="before")
    @classmethod
    def _coerce_stages(cls, v: Any) -> Any:
        return tuple(v) if isinstance(v, list) else v

    @field_validator("auxiliaries", mode="before")
    @classmethod
    def _coerce_auxiliaries(cls, v: Any) -> Any:
        if isinstance(v, list):
            return tuple(KDEntry(**x) if isinstance(x, dict) else x for x in v)
        return v

    @field_validator("scale")
    @classmethod
    def _valid_scale(cls, v: str) -> str:
        if v not in VALID_SCALES:
            raise ValueError(f"scale={v!r} not in {sorted(VALID_SCALES)}")
        return v

    @field_validator("conv_type")
    @classmethod
    def _valid_conv_type(cls, v: str) -> str:
        if v not in {"gatv2", "gat", "gps"}:
            raise ValueError("conv_type must be one of: gatv2, gat, gps")
        return v

    @field_validator("loss_fn")
    @classmethod
    def _valid_loss_fn(cls, v: str) -> str:
        if v not in {"focal", "ce", "weighted_ce"}:
            raise ValueError("loss_fn must be one of: focal, ce, weighted_ce")
        return v

    @field_validator("fusion_method")
    @classmethod
    def _valid_fusion_method(cls, v: str) -> str:
        if v not in VALID_FUSION_METHODS:
            raise ValueError(f"fusion_method={v!r} not in {sorted(VALID_FUSION_METHODS)}")
        return v

    @field_validator("model_type")
    @classmethod
    def _valid_model_type(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_MODEL_TYPES:
            raise ValueError(f"model_type={v!r} not in {sorted(VALID_MODEL_TYPES)}")
        return v

    @model_validator(mode="after")
    def _stages_exist(self) -> TrainingRunConfig:
        bad = [s for s in self.stages if s not in STAGES]
        if bad:
            raise ValueError(f"Unknown stages: {bad}. Valid: {sorted(STAGES)}")
        return self

    def merge(self, overrides: dict[str, Any]) -> TrainingRunConfig:
        return TrainingRunConfig(**{**self.model_dump(), **overrides})


class _SweepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_family: str
    stage: str
    scale: str | list[str] = "small"
    fusion_method: str | list[str] | None = None
    model_overrides: dict[str, Any] = Field(default_factory=dict)
    kd: KDEntry | None = None


class _SelectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[str] = Field(default_factory=list)
    model_families: list[str] = Field(default_factory=list)
    scales: list[str] = Field(default_factory=list)
    stages: dict[str, list[str]] = Field(default_factory=dict)
    fusion_methods: list[str] = Field(default_factory=list)


class _RecipeEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe: dict[str, Any] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    selection: _SelectionSpec | None = None
    sweeps: list[_SweepSpec] = Field(default_factory=list)
    trainer_overrides: dict[str, Any] = Field(default_factory=dict)
    stage_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    resource_overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stage_overrides")
    @classmethod
    def _valid_stage_names(cls, v: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        bad = [s for s in v if s not in STAGES]
        if bad:
            raise ValueError(f"Unknown stages in stage_overrides: {bad}. Valid: {sorted(STAGES)}")
        return v


def expand_recipe_configs(raw_recipe: dict[str, Any]) -> dict[str, Any]:
    """Normalize a rendered recipe dict to an orchestrator-ready config list.

    ``raw_recipe`` is the output of rendering a ``configs/recipes/*.jsonnet``
    file. Pydantic validates the envelope shape, then Jsonnet expands
    sweeps/selections via ``configs/recipes/_expand.jsonnet``.
    """
    envelope = _RecipeEnvelope(**raw_recipe)
    payload = envelope.model_dump(exclude_none=True)
    return render(
        CONFIG_DIR / "recipes" / "_expand.jsonnet",
        tla={
            "recipe": payload,
            "valid_scales": sorted(VALID_SCALES),
            "valid_fusion_methods": sorted(VALID_FUSION_METHODS),
        },
    )
