"""Config resolution and cross-field validation for pipeline runs.

``ConfigResolver`` is the single point that turns a ``StageConfig`` into a
fully rendered, validated ``ResolvedConfig``. Cross-field rules validate
the combined ``(TrainingSpec, ResourceSpec, StageConfig, rendered_dict)``
tuple after rendering — catching resource/config mismatches that neither
Pydantic structural validation nor jsonnet can see alone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from graphids.config.jsonnet import render
from graphids.config.schemas import ValidatedConfig, validate_config
from graphids.config.topology import TOPOLOGY, PathContext
from graphids.log import get_logger
from graphids.orchestrate.contracts import TrainingSpec, build_tla_dict
from graphids.orchestrate.planning import StageConfig
from graphids.slurm import ResourceSpec, apply_resource_overrides, get_resources

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cross-field validation rules
# ---------------------------------------------------------------------------

RuleFn = Callable[[TrainingSpec, ResourceSpec, StageConfig, dict[str, Any]], list[str]]
RulePred = Callable[[TrainingSpec, ResourceSpec, StageConfig, dict[str, Any]], bool]


@dataclass(frozen=True)
class ValidationRule:
    """One cross-field constraint applied during config resolution."""

    name: str
    severity: Literal["error", "warning"]
    applies: RulePred
    check: RuleFn


def _data_init(merged: dict[str, Any]) -> dict[str, Any]:
    return merged.get("data", {}).get("init_args", {}) or {}


def _always(spec, res, cfg, merged) -> bool:  # noqa: ARG001
    return True


def _is_gpu_stage(spec, res, cfg, merged) -> bool:  # noqa: ARG001
    return cfg.stage != "evaluation" and bool(res.gres)


def _is_supervised(spec, res, cfg, merged) -> bool:  # noqa: ARG001
    return cfg.stage == "supervised"


def _is_fusion_rl(spec, res, cfg, merged) -> bool:  # noqa: ARG001
    return cfg.stage == "fusion" and cfg.model_type in ("dqn", "bandit")


def _check_num_workers_within_cpus(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    max_workers = res.cpus_per_task - 1
    if res.num_workers > max_workers:
        return [f"num_workers={res.num_workers} exceeds cpus_per_task-1={max_workers}"]
    return []


def _check_rendered_num_workers_within_cpus(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    max_workers = res.cpus_per_task - 1
    rendered_workers = _data_init(merged).get("num_workers")
    if rendered_workers is not None and int(rendered_workers) > max_workers:
        return [
            f"data.init_args.num_workers={rendered_workers} in rendered config exceeds "
            f"cpus_per_task-1={max_workers} in resource profile"
        ]
    return []


def _check_gpu_partition_consistency(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    if "gpu" not in res.partition:
        return [f"gres={res.gres!r} set but partition={res.partition!r} is not a GPU partition"]
    return []


def _check_datamodule_epoch_sync(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    data_max = _data_init(merged).get("max_epochs")
    trainer_max = (merged.get("trainer") or {}).get("max_epochs")
    if data_max is not None and trainer_max is not None and int(data_max) != int(trainer_max):
        return [
            f"data.init_args.max_epochs={data_max} != "
            f"trainer.max_epochs={trainer_max} — difficulty ramp "
            f"will be scheduled over the wrong epoch count"
        ]
    return []


def _check_fusion_rl_batch_size_override(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    """RL fusion methods ignore batch_size (episode_sample_size controls it)."""
    tla_trainer = spec.jsonnet_tla.get("trainer_overrides", {}) or {}
    tla_stage = spec.jsonnet_tla.get("stage_overrides", {}) or {}
    if "data.init_args.batch_size" in tla_trainer or "data.init_args.batch_size" in tla_stage:
        return [
            f"batch_size override has no effect for RL fusion method "
            f"'{cfg.model_type}' — episode_sample_size controls batch size"
        ]
    return []


def _check_fusion_rl_batch_size_rendered(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    rendered_bs = _data_init(merged).get("batch_size")
    if rendered_bs is not None:
        return [
            f"data.init_args.batch_size={rendered_bs} has no effect for RL method "
            f"'{cfg.model_type}' — episode_sample_size controls batch size"
        ]
    return []


_RULES: tuple[ValidationRule, ...] = (
    ValidationRule(
        name="num_workers_within_cpus",
        severity="error",
        applies=_always,
        check=_check_num_workers_within_cpus,
    ),
    ValidationRule(
        name="rendered_num_workers_within_cpus",
        severity="error",
        applies=_always,
        check=_check_rendered_num_workers_within_cpus,
    ),
    ValidationRule(
        name="gpu_partition_consistency",
        severity="error",
        applies=_is_gpu_stage,
        check=_check_gpu_partition_consistency,
    ),
    ValidationRule(
        name="datamodule_epoch_sync",
        severity="error",
        applies=_is_supervised,
        check=_check_datamodule_epoch_sync,
    ),
    ValidationRule(
        name="fusion_rl_batch_size_override",
        severity="error",
        applies=_is_fusion_rl,
        check=_check_fusion_rl_batch_size_override,
    ),
    ValidationRule(
        name="fusion_rl_batch_size_rendered",
        severity="warning",
        applies=_is_fusion_rl,
        check=_check_fusion_rl_batch_size_rendered,
    ),
)


class _StageValidation(BaseModel):
    spec: TrainingSpec
    resources: ResourceSpec
    cfg: StageConfig
    merged: dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> _StageValidation:
        errors: list[str] = []
        warnings: list[tuple[str, str]] = []
        for rule in _RULES:
            if not rule.applies(self.spec, self.resources, self.cfg, self.merged):
                continue
            messages = rule.check(self.spec, self.resources, self.cfg, self.merged)
            if not messages:
                continue
            if rule.severity == "error":
                errors.extend(messages)
            else:
                warnings.extend((rule.name, msg) for msg in messages)

        for rule_name, msg in warnings:
            log.warning(
                "cross_field_warning",
                asset=self.cfg.asset_name,
                stage=self.cfg.stage,
                rule=rule_name,
                warning=msg,
            )

        if errors:
            raise ValueError("; ".join(errors))

        return self


def validate_stage_config(
    *,
    spec: TrainingSpec,
    resources: ResourceSpec,
    cfg: StageConfig,
    merged: dict[str, Any],
) -> None:
    """Run cross-field validation rules against the resolved config."""
    _StageValidation(spec=spec, resources=resources, cfg=cfg, merged=merged)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def _warn_stage_monitor_mismatch(validated: ValidatedConfig, stage: str, label: str) -> None:
    """Log a warning if a stage's monitor/mode diverges from its archetype."""
    family = TOPOLOGY.stage_family_map.get(stage)
    if family is None:
        return
    exp_monitor, exp_mode = ("val_acc", "max") if family == "fusion" else ("val_loss", "min")
    if validated.checkpoint_monitor != exp_monitor or validated.checkpoint_mode != exp_mode:
        log.warning(
            "stage_monitor_mismatch",
            asset=label,
            stage=stage,
            got=f"{validated.checkpoint_monitor}/{validated.checkpoint_mode}",
            expected=f"{exp_monitor}/{exp_mode}",
        )


@dataclass(frozen=True)
class OverrideRecord:
    """One override applied during resolution."""

    key: str
    value: str | int | float
    source: str  # recipe_trainer, recipe_resource, kd, stage_override
    stage: str | None = None  # None = all stages, else stage-scoped


@dataclass(frozen=True)
class ResolvedConfig:
    """Complete, validated output of ConfigResolver."""

    spec: TrainingSpec
    resources: ResourceSpec
    paths: PathContext
    audit: tuple[OverrideRecord, ...]
    validated: ValidatedConfig | None = None
    rendered: dict[str, Any] | None = None


class ConfigResolver:
    """Single merge point for all pipeline config resolution.

    Builds the typed TLA dict, renders jsonnet, runs Pydantic structural
    + cross-field validation, and emits an audit trail of every override.
    """

    def __init__(self, lake_root: str, user: str) -> None:
        self._lake_root = lake_root
        self._user = user

    def resolve(
        self,
        cfg: StageConfig,
        *,
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str] | None = None,
    ) -> ResolvedConfig:
        """Resolve a StageConfig into a validated TrainingSpec + ResourceSpec."""
        if upstream_ckpts is None:
            upstream_ckpts = {}
        audit: list[OverrideRecord] = []

        # --- Paths ---
        paths = PathContext(
            lake_root=self._lake_root,
            user=self._user,
            dataset=dataset,
            model_type=cfg.model_type,
            scale=cfg.scale,
            stage=cfg.stage,
            identity=cfg.identity,
            kd_tag=cfg.kd_tag,
            seed=seed,
        )

        # --- Audit trail (recipe overrides, KD, resources) ---
        if cfg.trainer_overrides:
            for k, v in cfg.trainer_overrides.items():
                audit.append(OverrideRecord(key=k, value=v, source="recipe_trainer"))

        if cfg.stage_overrides:
            for k, v in cfg.stage_overrides.items():
                audit.append(
                    OverrideRecord(key=k, value=v, source="stage_override", stage=cfg.stage)
                )

        if cfg.kd_overrides:
            audit.append(
                OverrideRecord(key="model.init_args.auxiliaries", value="<kd entry>", source="kd")
            )

        # --- Build TLA dict for jsonnet render ---
        tla = build_tla_dict(
            cfg,
            dataset=dataset,
            seed=seed,
            run_dir=str(paths.run_dir),
            upstream_ckpts=upstream_ckpts,
            upstream_model_families=cfg.upstream_model_families,
            kd_overrides=cfg.kd_overrides or None,
            trainer_overrides=cfg.trainer_overrides or None,
            stage_overrides=cfg.stage_overrides or None,
        )

        spec = TrainingSpec(
            stage=cfg.stage,
            model_family=cfg.model_type,
            scale=cfg.scale,
            dataset=dataset,
            seed=seed,
            run_dir=str(paths.run_dir),
            jsonnet_path=cfg.jsonnet_path,
            jsonnet_tla=tla,
            model_init_overrides=cfg.model_init_overrides,
            upstream_ckpt_paths=upstream_ckpts,
            upstream_model_families=cfg.upstream_model_families,
        )

        # --- Build ResourceSpec with overrides ---
        resources = get_resources(
            cfg.resource_model or cfg.model_type,
            cfg.scale,
            cfg.stage,
            dataset=dataset,
        )
        if cfg.resource_overrides:
            resources = apply_resource_overrides(resources, cfg.resource_overrides)
            for k, v in cfg.resource_overrides.items():
                audit.append(OverrideRecord(key=k, value=v, source="recipe_resource"))

        # --- Render the jsonnet chain ---
        rendered = render(spec.jsonnet_path, spec.jsonnet_tla)

        # --- Pydantic structural + convention validation ---
        try:
            validated = validate_config(rendered)
        except ValueError as e:
            raise ValueError(
                f"{cfg.asset_name} config validation (run_dir={paths.run_dir}): {e}"
            ) from e
        _warn_stage_monitor_mismatch(validated, cfg.stage, cfg.asset_name)

        # --- Cross-field validation ---
        try:
            validate_stage_config(spec=spec, resources=resources, cfg=cfg, merged=rendered)
        except ValueError as e:
            raise ValueError(f"{cfg.asset_name} cross-field validation: {e}") from e

        audit_tuple = tuple(audit)
        if audit_tuple:
            log.info(
                "config_resolved",
                asset=cfg.asset_name,
                dataset=dataset,
                seed=seed,
                overrides=[
                    {"key": r.key, "value": r.value, "source": r.source, "stage": r.stage}
                    for r in audit_tuple
                ],
            )

        return ResolvedConfig(
            spec=spec,
            resources=resources,
            paths=paths,
            audit=audit_tuple,
            validated=validated,
            rendered=rendered,
        )


__all__ = [
    "ConfigResolver",
    "OverrideRecord",
    "ResolvedConfig",
    "validate_stage_config",
]
