"""Exclusive config merge path for pipeline runs.

ConfigResolver subsumes the two separate override merge sites (trainer
overrides in execution.py, resource overrides in assets.py) into a single
validated resolution with cross-field checks and an audit trail.

Retry scaling (scale_resources on OOM/TIMEOUT) is a post-resolution runtime
concern — it stays in the asset closure, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog

from graphids.config import PathContext
from graphids.config.yaml_utils import merge_yaml_chain
from graphids.core.contracts import TrainingSpec
from graphids.orchestrate.planning import StageConfig
from graphids.slurm import ResourceSpec, apply_resource_overrides, get_resources

log = structlog.get_logger()


@dataclass(frozen=True)
class OverrideRecord:
    """One override applied during resolution."""

    key: str
    value: str | int | float
    source: str  # recipe_trainer, recipe_resource, kd, resume_ckpt


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

    @staticmethod
    def _merge_yaml_chain(
        config_files: tuple[str, ...], runtime_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge YAML config chain + runtime overrides for cross-field validation."""
        return merge_yaml_chain(config_files, runtime_overrides)

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

        if cfg.kd_overrides:
            key = "model.init_args.auxiliaries"
            val = json.dumps([cfg.kd_overrides])
            runtime_overrides[key] = val
            audit.append(OverrideRecord(key=key, value=val, source="kd"))

        if paths.last_ckpt_file.exists():
            runtime_overrides["ckpt_path"] = str(paths.last_ckpt_file)
            audit.append(OverrideRecord(
                key="ckpt_path", value=str(paths.last_ckpt_file),
                source="resume_ckpt",
            ))

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
        merged_yaml = self._merge_yaml_chain(cfg.config_files, runtime_overrides)

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
                    {"key": r.key, "value": r.value, "source": r.source}
                    for r in audit_tuple
                ],
            )

        return ResolvedConfig(
            spec=spec,
            resources=resources,
            paths=paths,
            audit=audit_tuple,
        )

    def _validate_cross_fields(
        self,
        spec: TrainingSpec,
        resources: ResourceSpec,
        cfg: StageConfig,
        merged_yaml: dict[str, Any],
    ) -> None:
        """Validate constraints that span TrainingSpec, ResourceSpec, and YAML configs."""
        errors: list[str] = []
        data_init = merged_yaml.get("data", {}).get("init_args", {})
        trainer = merged_yaml.get("trainer", {})

        # --- Resource-level checks ---

        # num_workers must fit within allocated CPUs (leave 1 for main process)
        max_workers = resources.cpus_per_task - 1
        if resources.num_workers > max_workers:
            errors.append(
                f"num_workers={resources.num_workers} exceeds "
                f"cpus_per_task-1={max_workers}"
            )

        # YAML num_workers vs resource profile CPUs
        yaml_workers = data_init.get("num_workers")
        if yaml_workers is not None and int(yaml_workers) > max_workers:
            errors.append(
                f"data.init_args.num_workers={yaml_workers} in YAML exceeds "
                f"cpus_per_task-1={max_workers} in resource profile"
            )

        # GPU required for training stages (partition must be gpu/gpudebug)
        if cfg.stage != "evaluation" and resources.gres:
            if "gpu" not in resources.partition:
                errors.append(
                    f"gres={resources.gres!r} set but partition="
                    f"{resources.partition!r} is not a GPU partition"
                )

        # --- YAML-aware checks ---

        # Curriculum epoch sync: data module max_epochs must match trainer max_epochs
        if cfg.stage == "curriculum":
            data_max_epochs = data_init.get("max_epochs")
            trainer_max_epochs = trainer.get("max_epochs")
            if (
                data_max_epochs is not None
                and trainer_max_epochs is not None
                and int(data_max_epochs) != int(trainer_max_epochs)
            ):
                errors.append(
                    f"CurriculumDataModule.max_epochs={data_max_epochs} != "
                    f"trainer.max_epochs={trainer_max_epochs} — curriculum "
                    f"difficulty ramp will be scheduled over the wrong epoch count"
                )

        # Fusion RL methods ignore batch_size
        if cfg.stage == "fusion" and cfg.model_type in ("dqn", "bandit"):
            if "data.init_args.batch_size" in spec.runtime_overrides:
                errors.append(
                    f"batch_size override has no effect for RL fusion method "
                    f"'{cfg.model_type}' — episode_sample_size controls batch size"
                )
            yaml_bs = data_init.get("batch_size")
            if yaml_bs is not None:
                log.warning(
                    "dead_config",
                    key="data.init_args.batch_size",
                    value=yaml_bs,
                    reason=f"RL method '{cfg.model_type}' uses episode_sample_size",
                )

        if errors:
            raise ValueError(
                f"Cross-field validation failed for {cfg.asset_name}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
