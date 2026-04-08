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

from typing import Annotated, Any, Literal  # noqa: F401 (resolved by model_rebuild)

from pydantic import (  # noqa: F401 (AfterValidator resolved by model_rebuild)
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from graphids.config.constants import (
    CONFIG_DIR,
    VALID_FUSION_METHODS,
    VALID_SCALES,
    ModelType,  # noqa: F401 (resolved by model_rebuild)
)
from graphids.config.jsonnet import render
from graphids.config.topology import TOPOLOGY  # noqa: F401 (used in field default)


def check_in(valid, label):  # noqa: F401 (resolved by model_rebuild)
    def _v(v):
        if v not in valid:
            raise ValueError(f"{label}={v!r} not in {sorted(valid)}")
        return v

    return _v


# Type aliases — evaluated eagerly, immune to __future__.annotations quote-stripping.
_ConvType = Literal["gatv2", "gat", "gps"]
_LossFn = Literal["focal", "ce", "weighted_ce"]


class KDEntry(BaseModel):
    """Recipe-side KD config schema — superset of ``KDAuxiliary``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["kd"] = "kd"
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    teacher_config: str | None = None
    teacher_scale: Annotated[str, AfterValidator(check_in(VALID_SCALES, "teacher_scale"))] = "large"
    temperature: float | None = Field(default=None, gt=0.0)
    model_path: str | None = None
    vgae_latent_weight: float | None = None
    vgae_recon_weight: float | None = None


class TrainingRunConfig(BaseModel):
    """Typed boundary input for a training run identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stages: tuple[Annotated[str, AfterValidator(check_in(TOPOLOGY.stages, "stage"))], ...] = tuple(
        TOPOLOGY.default_stages
    )
    scale: Annotated[str, AfterValidator(check_in(VALID_SCALES, "scale"))] = "small"
    conv_type: _ConvType = "gatv2"
    loss_fn: _LossFn = "focal"
    fusion_method: Annotated[
        str, AfterValidator(check_in(VALID_FUSION_METHODS, "fusion_method"))
    ] = "bandit"
    variational: bool = True
    model_type: ModelType | None = None
    auxiliaries: tuple[KDEntry, ...] = ()

    @field_validator("stages", mode="before")
    @classmethod
    def _coerce_stages(cls, v: Any) -> Any:
        return tuple(v) if isinstance(v, list) else v

    @field_validator("auxiliaries", mode="before")
    @classmethod
    def _coerce_auxiliaries(cls, v: Any) -> Any:
        if isinstance(v, list):
            return tuple(KDEntry.model_validate(x) if isinstance(x, dict) else x for x in v)
        return v

    def merge(self, overrides: dict[str, Any]) -> TrainingRunConfig:
        return TrainingRunConfig.model_validate({**self.model_dump(), **overrides})


# Resolve deferred Annotated/Literal annotations (from __future__ import annotations).
KDEntry.model_rebuild()
TrainingRunConfig.model_rebuild()


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
        bad = [s for s in v if s not in TOPOLOGY.stages]
        if bad:
            raise ValueError(
                f"Unknown stages in stage_overrides: {bad}. Valid: {sorted(TOPOLOGY.stages)}"
            )
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
