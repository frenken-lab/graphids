"""Exclusive config merge path for pipeline runs.

ConfigResolver subsumes the two separate override merge sites (trainer
overrides in execution.py, resource overrides in assets.py) into a single
validated resolution with cross-field checks and an audit trail.

``validate_cli_chain`` runs the resolved spec through the full jsonargparse
schema + convention checks, so override-key typos, null list fields, and
logger/callback wiring mismatches die at planning time (ADR 0009).
"""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import yaml

from graphids.log import get_logger

from graphids.config import PathContext
from graphids.config.yaml_utils import merge_yaml_chain
from graphids.core.contracts import TrainingContract, TrainingSpec
from graphids.orchestrate.planning import StageConfig
from graphids.slurm import ResourceSpec, apply_resource_overrides, get_resources

log = get_logger(__name__)

# Fusion stages optimize val_acc/max; all others val_loss/min.
_STAGE_MONITORS = {
    "autoencoder": ("val_loss", "min"),
    "normal": ("val_loss", "min"),
    "curriculum": ("val_loss", "min"),
    "fusion": ("val_acc", "max"),
}


# ---------------------------------------------------------------------------
# Cross-field validation rules
#
# Each constraint that spans TrainingSpec + ResourceSpec + merged YAML lives
# as a single ValidationRule. `applies` gates the rule (so curriculum-only
# checks don't run on other stages); `check` returns the violation messages
# or an empty list. Severity decides error-vs-warning at the call site.
# Adding a rule = adding one entry to _RULES. Each rule is independently
# unit-testable (see tests/orchestrate/test_overrides.py::TestValidationRules).
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


def _is_curriculum(spec, res, cfg, merged) -> bool:  # noqa: ARG001
    return cfg.stage == "curriculum"


def _is_fusion_rl(spec, res, cfg, merged) -> bool:  # noqa: ARG001
    return cfg.stage == "fusion" and cfg.model_type in ("dqn", "bandit")


def _check_num_workers_within_cpus(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    max_workers = res.cpus_per_task - 1
    if res.num_workers > max_workers:
        return [
            f"num_workers={res.num_workers} exceeds "
            f"cpus_per_task-1={max_workers}"
        ]
    return []


def _check_yaml_num_workers_within_cpus(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    max_workers = res.cpus_per_task - 1
    yaml_workers = _data_init(merged).get("num_workers")
    if yaml_workers is not None and int(yaml_workers) > max_workers:
        return [
            f"data.init_args.num_workers={yaml_workers} in YAML exceeds "
            f"cpus_per_task-1={max_workers} in resource profile"
        ]
    return []


def _check_gpu_partition_consistency(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    if "gpu" not in res.partition:
        return [
            f"gres={res.gres!r} set but partition="
            f"{res.partition!r} is not a GPU partition"
        ]
    return []


def _check_curriculum_epoch_sync(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    data_max = _data_init(merged).get("max_epochs")
    trainer_max = (merged.get("trainer") or {}).get("max_epochs")
    if (
        data_max is not None
        and trainer_max is not None
        and int(data_max) != int(trainer_max)
    ):
        return [
            f"CurriculumDataModule.max_epochs={data_max} != "
            f"trainer.max_epochs={trainer_max} — curriculum "
            f"difficulty ramp will be scheduled over the wrong epoch count"
        ]
    return []


def _check_fusion_rl_batch_size_override(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    if "data.init_args.batch_size" in spec.runtime_overrides:
        return [
            f"batch_size override has no effect for RL fusion method "
            f"'{cfg.model_type}' — episode_sample_size controls batch size"
        ]
    return []


def _check_fusion_rl_batch_size_yaml(spec, res, cfg, merged) -> list[str]:  # noqa: ARG001
    yaml_bs = _data_init(merged).get("batch_size")
    if yaml_bs is not None:
        return [
            f"data.init_args.batch_size={yaml_bs} has no effect for RL method "
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
        name="yaml_num_workers_within_cpus",
        severity="error",
        applies=_always,
        check=_check_yaml_num_workers_within_cpus,
    ),
    ValidationRule(
        name="gpu_partition_consistency",
        severity="error",
        applies=_is_gpu_stage,
        check=_check_gpu_partition_consistency,
    ),
    ValidationRule(
        name="curriculum_epoch_sync",
        severity="error",
        applies=_is_curriculum,
        check=_check_curriculum_epoch_sync,
    ),
    ValidationRule(
        name="fusion_rl_batch_size_override",
        severity="error",
        applies=_is_fusion_rl,
        check=_check_fusion_rl_batch_size_override,
    ),
    ValidationRule(
        name="fusion_rl_batch_size_yaml",
        severity="warning",
        applies=_is_fusion_rl,
        check=_check_fusion_rl_batch_size_yaml,
    ),
)


def _convention_errors(dumped: dict, stage: str, label: str) -> list[str]:
    """Return fatal convention errors; emit non-fatal warnings via log."""
    errors: list[str] = []
    trainer = dumped.get("trainer") or {}
    logger_on = trainer.get("logger", True) is not False
    for cb in trainer.get("callbacks") or []:
        cp = cb.get("class_path", "")
        if "LearningRateMonitor" in cp and not logger_on:
            errors.append(f"{cp.rsplit('.', 1)[-1]} requires trainer.logger=true")
    model_args = dumped.get("model", {}).get("init_args") or {}
    for fld in ("pool_aggrs", "hidden_dims", "auxiliaries"):
        if fld in model_args and model_args[fld] is None:
            errors.append(f"model.init_args.{fld} is null")
    exp = _STAGE_MONITORS.get(stage)
    if exp:
        for ns in ("checkpoint", "early_stopping"):
            cfg = dumped.get(ns) or {}
            if not isinstance(cfg, dict):
                continue
            for field, expected in zip(("monitor", "mode"), exp):
                val = cfg.get(field)
                if val is not None and val != expected:
                    log.warning(
                        "convention_mismatch",
                        asset=label, ns=ns, field=field,
                        got=val, expected=expected,
                    )
    return errors


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


class ConfigResolver:
    """Single merge point for all pipeline config resolution.

    Replaces the separate merge sites in execution.py (training_spec) and
    assets.py (apply_resource_overrides). All overrides are applied here,
    cross-field constraints validated, and an audit trail emitted.
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
        upstream_ckpts: dict[str, str],
    ) -> ResolvedConfig:
        """Resolve a StageConfig into a validated TrainingSpec + ResourceSpec."""
        audit: list[OverrideRecord] = []

        # --- Paths ---
        paths = PathContext(
            lake_root=self._lake_root, user=self._user, dataset=dataset,
            model_type=cfg.model_type, scale=cfg.scale, stage=cfg.stage,
            identity=cfg.identity, kd_tag=cfg.kd_tag, seed=seed,
        )

        # --- Build TrainingSpec with all overrides merged ---
        runtime_overrides: dict[str, Any] = {}

        if cfg.trainer_overrides:
            runtime_overrides.update(cfg.trainer_overrides)
            for k, v in cfg.trainer_overrides.items():
                audit.append(OverrideRecord(key=k, value=v, source="recipe_trainer"))

        if cfg.stage_overrides:
            runtime_overrides.update(cfg.stage_overrides)
            for k, v in cfg.stage_overrides.items():
                audit.append(OverrideRecord(key=k, value=v, source="stage_override", stage=cfg.stage))

        if cfg.kd_overrides:
            key = "model.init_args.auxiliaries"
            val = json.dumps([cfg.kd_overrides])
            runtime_overrides[key] = val
            audit.append(OverrideRecord(key=key, value=val, source="kd"))

        # Auto-resume from last.ckpt is handled at training time in
        # train_entrypoint.py — NOT here. The orchestrator runs on a different
        # node (NFS-cached); checking exists() here creates a race condition.

        spec = TrainingSpec(
            stage=cfg.stage,
            model_family=cfg.model_type,
            scale=cfg.scale,
            dataset=dataset,
            seed=seed,
            run_dir=str(paths.run_dir),
            config_files=cfg.config_files,
            model_init_overrides=cfg.model_init_overrides,
            upstream_ckpt_paths=upstream_ckpts,
            upstream_model_families=cfg.upstream_model_families,
            runtime_overrides=runtime_overrides,
        )

        # --- Build ResourceSpec with overrides ---
        resources = get_resources(
            cfg.resource_model or cfg.model_type, cfg.scale, cfg.stage,
        )
        if cfg.resource_overrides:
            resources = apply_resource_overrides(resources, cfg.resource_overrides)
            for k, v in cfg.resource_overrides.items():
                audit.append(OverrideRecord(key=k, value=v, source="recipe_resource"))

        # --- Merge YAML chain for cross-field validation ---
        merged_yaml = merge_yaml_chain(cfg.config_files, runtime_overrides)

        # --- Cross-field validation ---
        self._validate_cross_fields(spec, resources, cfg, merged_yaml)

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
        )

    def resolve_and_validate(
        self,
        cfg: StageConfig,
        *,
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str] | None = None,
    ) -> "ResolvedConfig":
        """Resolve + pre-validate in one call. Used by assets.py and validate.py."""
        resolved = self.resolve(
            cfg, dataset=dataset, seed=seed, upstream_ckpts=upstream_ckpts or {},
        )
        self.validate_cli_chain(resolved.spec)
        return resolved

    def validate_cli_chain(self, spec: TrainingSpec) -> None:
        """Pre-validate spec through jsonargparse schema + convention checks (ADR 0009)."""
        from graphids._lightning import schema_parser  # lazy torch import

        parser = schema_parser()
        merged = merge_yaml_chain(
            spec.config_files, TrainingContract.to_override_dict(spec),
        )
        label = f"{spec.stage}/{spec.model_family}_{spec.scale}"

        # jsonargparse calls sys.exit() on parse errors; capture stderr + catch.
        err_buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(err_buf):
                parsed = parser.parse_object(merged)
        except (Exception, SystemExit) as e:
            msg = next(
                (ln for ln in reversed(err_buf.getvalue().splitlines()) if ln.strip()),
                str(e),
            )
            raise ValueError(f"{label} schema (run_dir={spec.run_dir}): {msg}") from e

        dumped = yaml.safe_load(
            parser.dump(parsed, skip_link_targets=False, skip_none=False)
        )
        errors = _convention_errors(dumped, spec.stage, label)
        if errors:
            raise ValueError(f"{label} conventions: " + "; ".join(errors))

    def _validate_cross_fields(
        self,
        spec: TrainingSpec,
        resources: ResourceSpec,
        cfg: StageConfig,
        merged_yaml: dict[str, Any],
    ) -> None:
        """Validate constraints that span TrainingSpec, ResourceSpec, and YAML configs.

        Thin dispatcher over _RULES. Each rule is a ValidationRule with its
        own `applies` predicate and `check` function; errors are collected
        and raised as a single ValueError, warnings are logged and do not
        fail resolution. See the _RULES tuple at module top for the full list.
        """
        errors: list[str] = []
        for rule in _RULES:
            if not rule.applies(spec, resources, cfg, merged_yaml):
                continue
            messages = rule.check(spec, resources, cfg, merged_yaml)
            if not messages:
                continue
            if rule.severity == "error":
                errors.extend(messages)
            else:
                for msg in messages:
                    log.warning("convention_mismatch", rule=rule.name, msg=msg)
        if errors:
            raise ValueError(
                f"Cross-field validation failed for {cfg.asset_name}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
