"""Cross-field validation rules for resolved stage configs.

Orchestration-level validation: rules that fire against the combined
``(TrainingSpec, ResourceSpec, StageConfig, rendered_dict)`` tuple after
``ConfigResolver`` has built the spec and rendered the jsonnet chain.
This is *not* rendered-dict structural validation (that lives in
``graphids.config.schemas.ValidatedConfig``); it's cross-layer checks
that require the SLURM resource spec and the orchestrator's stage config
to validate against the rendered config.

Both the rule engine and the Pydantic ``StageValidation`` entry point
live here so the whole orchestration-side validation stack is in one
module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from graphids.log import get_logger
from graphids.orchestrate.contracts import TrainingSpec
from graphids.orchestrate.planning import StageConfig
from graphids.slurm.resources import ResourceSpec

RuleFn = Callable[[TrainingSpec, ResourceSpec, StageConfig, dict[str, Any]], list[str]]
RulePred = Callable[[TrainingSpec, ResourceSpec, StageConfig, dict[str, Any]], bool]

log = get_logger(__name__)


@dataclass(frozen=True)
class ValidationRule:
    """One cross-field constraint applied during config resolution."""

    name: str
    severity: Literal["error", "warning"]
    applies: RulePred
    check: RuleFn


# -----------------------------------------------------------------------------
# Predicates
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Pydantic gate
# -----------------------------------------------------------------------------


class StageValidation(BaseModel):
    spec: TrainingSpec
    resources: ResourceSpec
    cfg: StageConfig
    merged: dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> StageValidation:
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
            joined = "; ".join(errors)
            raise ValueError(joined)

        return self


def validate_stage_config(
    *,
    spec: TrainingSpec,
    resources: ResourceSpec,
    cfg: StageConfig,
    merged: dict[str, Any],
) -> StageValidation:
    """Run Pydantic validation for cross-field rules."""
    return StageValidation(
        spec=spec,
        resources=resources,
        cfg=cfg,
        merged=merged,
    )
