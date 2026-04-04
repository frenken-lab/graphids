"""Typed config contracts and recipe expansion facade."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .topology import STAGES, VALID_FUSION_METHODS, VALID_MODEL_TYPES, VALID_SCALES


class KDEntry(BaseModel):
    """Unified KD config schema. Fields must match KDAuxiliary in core/models/_training.py.

    Teacher identification:
    - ``teacher_config`` (orchestration): names a recipe config by key. Planning
      wires this config as an upstream dependency, guaranteeing the student
      loads a specific teacher regardless of recipe key order. Required for
      pipeline runs; silent scale-based inference was removed (see ADR/risk
      doc — ordered iteration made it position-dependent).
    - ``teacher_scale`` (dev path): used by ``prepare_kd`` when running
      ``python -m graphids fit`` without the orchestrator, to recompute a
      deterministic checkpoint path. Ignored by planning.
    - ``model_path``: explicit teacher checkpoint path. Overrides everything
      else at runtime.
    """

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

    stages: tuple[str, ...] = ("autoencoder", "curriculum", "fusion")
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
    def _stages_exist(self) -> "TrainingRunConfig":
        bad = [s for s in self.stages if s not in STAGES]
        if bad:
            raise ValueError(f"Unknown stages: {bad}. Valid: {sorted(STAGES)}")
        return self

    def merge(self, overrides: dict[str, Any]) -> "TrainingRunConfig":
        return TrainingRunConfig(**{**self.model_dump(), **overrides})


def expand_recipe_configs(raw_recipe: dict[str, Any]) -> dict[str, Any]:
    """Normalize recipe envelopes to orchestrator-ready config lists."""
    from .recipe_expand import expand_recipe_configs as _impl

    return _impl(
        raw_recipe,
        valid_scales=VALID_SCALES,
        valid_fusion_methods=VALID_FUSION_METHODS,
    )
