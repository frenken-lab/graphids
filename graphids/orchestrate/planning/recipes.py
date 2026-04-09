"""Recipe expansion and planning models.

``expand_recipe_configs`` renders a recipe jsonnet through the expansion
pipeline (``configs/recipes/_expand.jsonnet``). ``TrainingRunConfig`` is
the typed view of the expanded ``defaults`` block used by
``enumerate_assets`` when building StageConfigs.
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
    FAMILY_FOR_MODEL_TYPE,
    VALID_FUSION_METHODS,
    VALID_SCALES,
    ModelType,  # noqa: F401 (resolved by model_rebuild)
)
from graphids.config.jsonnet import render
from graphids.config.topology import TOPOLOGY  # noqa: F401 (used in field default)

# identity key → recipe field name (where topology and recipe names differ)
_IDENTITY_TO_RECIPE: dict[str, str] = {"method": "fusion_method"}

# family → default model_type (first model_type for each family)
_DEFAULT_MODEL_TYPE: dict[str, str] = {}
for _mt, _fam in FAMILY_FOR_MODEL_TYPE.items():
    _DEFAULT_MODEL_TYPE.setdefault(_fam, _mt)


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

    def identity_for(self, stage: str) -> dict[str, Any]:
        """Identity key values for a stage, mapped from recipe fields."""
        stage_def = TOPOLOGY.stages[stage]
        result: dict[str, Any] = {}
        for key in stage_def.identity_keys:
            val = getattr(self, _IDENTITY_TO_RECIPE.get(key, key), None)
            if key == "model_type" and val is None:
                val = _DEFAULT_MODEL_TYPE.get(stage_def.family, "vgae")
            result[key] = val
        return result


# Resolve deferred Annotated/Literal annotations (from __future__ import annotations).
KDEntry.model_rebuild()
TrainingRunConfig.model_rebuild()


def expand_recipe_configs(raw_recipe: dict[str, Any]) -> dict[str, Any]:
    """Expand a rendered recipe dict to an orchestrator-ready config list.

    ``raw_recipe`` is the output of rendering a ``configs/recipes/*.jsonnet``
    file. Jsonnet expansion (``configs/recipes/_expand.jsonnet``) handles
    sweep/selection cartesian products, override flattening, and defaults.
    """
    return render(
        CONFIG_DIR / "recipes" / "_expand.jsonnet",
        tla={
            "recipe": raw_recipe,
            "valid_scales": sorted(VALID_SCALES),
            "valid_fusion_methods": sorted(VALID_FUSION_METHODS),
        },
    )
