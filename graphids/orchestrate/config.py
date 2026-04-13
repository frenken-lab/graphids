"""Orchestrate data types — Layer 0 of the orchestrate stack.

Every frozen Pydantic/dataclass type used by the planning, resolution,
instantiation, stage, and run layers lives here. No side effects, no
torch imports, no jsonnet subprocess — callers below this file produce
these types; callers above consume them.

The central boundary type is ``StageConfig``. It carries everything the
planner knows about one asset plus the ``to_tla_dict`` projection onto
the jsonnet TLA dict that every stage function consumes. Adding a new
TLA means editing ``to_tla_dict`` + the stage jsonnet signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal  # noqa: F401 (resolved by model_rebuild)

from pydantic import (  # noqa: F401 (AfterValidator resolved by model_rebuild)
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
)

from graphids.config.constants import (  # noqa: F401 (resolved by model_rebuild)
    FAMILY_FOR_MODEL_TYPE,
    PIPELINE_DEFAULTS,
    VALID_FUSION_METHODS,
    VALID_SCALES,
    ModelType,
)
from graphids.config.topology import TOPOLOGY

if TYPE_CHECKING:
    import torch.nn as nn

    from graphids.config.schemas import ValidatedConfig
    from graphids.core.trainer import Trainer

_D = PIPELINE_DEFAULTS
_UNSUPERVISED_MODELS = frozenset(k for k, v in FAMILY_FOR_MODEL_TYPE.items() if v == "unsupervised")

# identity key → recipe field name (where topology and recipe names differ)
_IDENTITY_TO_RECIPE: dict[str, str] = {"method": "fusion_method"}

# family → default model_type (first model_type for each family)
_DEFAULT_MODEL_TYPE: dict[str, str] = {}
for _mt, _fam in FAMILY_FOR_MODEL_TYPE.items():
    _DEFAULT_MODEL_TYPE.setdefault(_fam, _mt)


def check_in(valid, label):
    def _v(v):
        if v not in valid:
            raise ValueError(f"{label}={v!r} not in {sorted(valid)}")
        return v

    return _v


def check_all_in(valid, label):
    def _v(v):
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(f"Unknown {label}(s): {bad}. Valid: {sorted(valid)}")
        return v

    return _v


# Type aliases — evaluated eagerly, immune to __future__.annotations quote-stripping.
_ConvType = Literal["gatv2", "gat", "gps"]
_LossFn = Literal["focal", "ce", "weighted_ce"]


# ---------------------------------------------------------------------------
# Recipe-side schemas
# ---------------------------------------------------------------------------


class KDEntry(BaseModel):
    """KD auxiliary config schema — one entry per teacher."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["kd"] = "kd"
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
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


# ---------------------------------------------------------------------------
# StageConfig — the central planner output / resolver input
# ---------------------------------------------------------------------------


class StageConfig(BaseModel):
    """Training config for one asset. Pure data, no torch imports.

    ``to_tla_dict`` is the single place field names map to jsonnet TLA
    keys. Adding a new TLA means editing this method + the stage jsonnet
    signature.
    """

    model_config = ConfigDict(frozen=True)

    stage: str
    model_type: str
    scale: str
    model_init_overrides: dict[str, Any] = Field(default_factory=dict)
    identity: str = ""
    resource_model: str = ""  # model key for resource lookup (fusion method for fusion stages)
    kd_overrides: dict[str, Any] = Field(default_factory=dict)
    trainer_overrides: dict[str, Any] = Field(default_factory=dict)
    stage_overrides: dict[str, Any] = Field(default_factory=dict)
    resource_overrides: dict[str, str | int] = Field(default_factory=dict)
    upstream_asset_names: tuple[str, ...] = ()
    upstream_model_families: dict[str, str] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def kd_tag(self) -> str:
        return "_kd" if self.kd_overrides else ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def asset_name(self) -> str:
        return f"{self.stage}{self.identity}{self.kd_tag}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def jsonnet_path(self) -> str:
        from graphids.orchestrate.planning import resolve_jsonnet_path

        return resolve_jsonnet_path(self.stage)

    def to_tla_dict(
        self,
        *,
        dataset: str,
        seed: int,
        run_dir: str,
        upstream_ckpts: dict[str, str],
        ckpt_path: str | None = None,
    ) -> dict[str, Any]:
        """Pack this StageConfig + runtime context into the jsonnet TLA dict.

        This is the ONLY place field names map to jsonnet TLA keys.
        Adding a new TLA means editing this method + the stage jsonnet
        signature.
        """
        tla: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "run_dir": run_dir,
            "scale": self.scale,
            "trainer_overrides": dict(self.trainer_overrides),
            "stage_overrides": dict(self.stage_overrides),
        }
        tla.update(self.model_init_overrides)

        stage_def = TOPOLOGY.stages.get(self.stage)
        accepted = set(stage_def.stage_tlas) if stage_def else set()

        if "fusion_method" in accepted:
            tla["fusion_method"] = self.resource_model or self.model_type

        for upstream_asset, ckpt in upstream_ckpts.items():
            family = self.upstream_model_families.get(upstream_asset)
            if family == "unsupervised" and "vgae_ckpt_path" in accepted:
                tla["vgae_ckpt_path"] = ckpt
            elif family == "supervised" and "gat_ckpt_path" in accepted:
                tla["gat_ckpt_path"] = ckpt

        if "distillation_config" in accepted:
            tla["distillation_config"] = dict(self.kd_overrides) if self.kd_overrides else None

        if ckpt_path is not None and "ckpt_path" in accepted:
            tla["ckpt_path"] = ckpt_path

        return tla

    @classmethod
    def for_stage(
        cls,
        stage: str,
        merged: TrainingRunConfig,
        *,
        upstream_names: list[str],
        upstream_models: dict[str, str],
        trainer_overrides: dict[str, Any] | None = None,
        stage_overrides: dict[str, Any] | None = None,
        resource_overrides: dict[str, str | int] | None = None,
    ) -> StageConfig:
        """Build a StageConfig from a merged training-run config + topology."""
        from graphids.config.topology import compute_identity_hash

        stage_def = TOPOLOGY.stages[stage]

        model_type = (
            merged.model_type
            if merged.model_type
            and stage_def.learning_type == "unsupervised"
            and merged.model_type in _UNSUPERVISED_MODELS
            else stage_def.family
        )

        id_cfg = merged.identity_for(stage)
        accepted = set(stage_def.stage_tlas)

        return cls(
            stage=stage,
            model_type=model_type,
            scale=merged.scale,
            model_init_overrides={
                k: v for k, v in id_cfg.items() if v is not None and k in accepted
            },
            identity=compute_identity_hash(stage, id_cfg),
            resource_model=merged.fusion_method if stage == "fusion" else model_type,
            kd_overrides=(
                merged.auxiliaries[0].model_dump(exclude_none=True) if merged.auxiliaries else {}
            ),
            trainer_overrides=trainer_overrides or {},
            stage_overrides=stage_overrides or {},
            resource_overrides=resource_overrides or {},
            upstream_asset_names=tuple(sorted(upstream_names)),
            upstream_model_families=upstream_models,
        )


# ---------------------------------------------------------------------------
# PipelineConfig — CLI-facing input schema
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """What to run in a single ``pipeline-run`` invocation."""

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
    conv_type: _ConvType = _D.get("conv_type", "gatv2")
    variational: bool = _D.get("variational", True)
    loss_fn: _LossFn = _D.get("loss_fn", "focal")
    tla_overrides: dict[str, Any] = Field(default_factory=dict)
    max_retries: int = 2

    def to_training_run(self) -> TrainingRunConfig:
        """Convert CLI fields to a planner-ready ``TrainingRunConfig``."""
        return TrainingRunConfig(
            stages=tuple(self.stages),
            scale=self.scale,
            conv_type=self.conv_type,
            variational=self.variational,
            loss_fn=self.loss_fn,
            fusion_method=self.fusion_method,
        )


# ---------------------------------------------------------------------------
# ResolvedConfig / InstantiatedRun / PipelineResult — later-layer outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfig:
    """Rendered, validated config ready for instantiation.

    Stage primitives (``orchestrate/stage.py``) consume this directly:
    they read ``rendered`` + ``validated`` for building trainer/model,
    ``run_dir`` / ``ckpt_file`` for marker + OTel export wiring, and
    ``stage_name`` for log fields. ``run_dir`` is ``None`` only for
    smoke invocations of the Typer CLI with no ``default_root_dir``
    set — markers and file exporters are skipped in that case.
    """

    rendered: dict[str, Any]
    validated: "ValidatedConfig"
    stage_name: str
    run_dir: Path | None
    ckpt_file: Path | None

    @classmethod
    def from_rendered(cls, rendered: dict[str, Any], *, stage_name: str) -> ResolvedConfig:
        """Validate a pre-rendered dict and pull ``run_dir`` from jsonnet.

        Used by the Typer CLI where the jsonnet is rendered from a file
        path directly (no ``StageConfig`` / ``PathContext``). ``run_dir``
        / ``ckpt_file`` come from ``trainer.default_root_dir``; both are
        ``None`` for smoke invocations that don't set a run_dir.
        """
        from graphids.config.constants import CKPT_SUBPATH
        from graphids.config.schemas import validate_config

        validated = validate_config(rendered)
        default_root = (rendered.get("trainer") or {}).get("default_root_dir") or ""
        run_dir = Path(default_root) if default_root else None
        ckpt_file = run_dir / CKPT_SUBPATH if run_dir else None
        return cls(
            rendered=rendered,
            validated=validated,
            stage_name=stage_name,
            run_dir=run_dir,
            ckpt_file=ckpt_file,
        )


@dataclass
class InstantiatedRun:
    """A wired (trainer, model, datamodule) triple built from a rendered config."""

    trainer: "Trainer"
    model: "nn.Module"
    datamodule: Any


@dataclass(frozen=True)
class PipelineResult:
    """Composite result of a full pipeline run."""

    checkpoints: dict[str, str]  # asset_name -> ckpt path
    analyzed_assets: list[str]  # asset_names whose analyzer succeeded
    stage_to_asset: dict[str, str]  # stage -> asset_name

    def checkpoints_by_stage(self) -> dict[str, str]:
        return {
            stage: self.checkpoints.get(asset, "")
            for stage, asset in self.stage_to_asset.items()
        }


# Resolve deferred Annotated/Literal annotations (from __future__ import annotations).
KDEntry.model_rebuild()
TrainingRunConfig.model_rebuild()
StageConfig.model_rebuild()
PipelineConfig.model_rebuild()
