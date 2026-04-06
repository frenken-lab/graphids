"""Exclusive config merge path for pipeline runs.

ConfigResolver is the single point that turns a ``StageConfig`` + its
planner-derived overrides into a fully rendered, validated
``ResolvedConfig``. It builds the typed TLA dict, calls
``render_config(spec.jsonnet_path, spec.jsonnet_tla)``, runs the Pydantic
structural gate (``graphids.config.validate_config``), runs cross-field
rules against the combined ``(spec, resources, stage_cfg, merged)``
tuple, and emits an audit trail of every override applied.
"""

from __future__ import annotations

from dataclasses import dataclass

from graphids.config.jsonnet import render
from graphids.config.schemas import (
    PathContext,
    ValidatedConfig,
    validate_config,
)
from graphids.log import get_logger
from graphids.orchestrate.contracts import TrainingSpec, build_tla_dict
from graphids.orchestrate.planning import StageConfig
from graphids.orchestrate.resolve.cross_field import validate_stage_config
from graphids.slurm import ResourceSpec, apply_resource_overrides, get_resources

log = get_logger(__name__)

# Fusion stages optimize val_acc/max; all others val_loss/min. Used for the
# stage-convention warning in `_warn_stage_monitor_mismatch` — a divergence
# is not fatal (``ValidatedConfig`` already forces checkpoint/early_stopping
# to agree) but a stage whose monitors don't match its archetype is almost
# always a mistake that should surface in the orchestrator logs.
_STAGE_MONITORS = {
    "autoencoder": ("val_loss", "min"),
    "supervised": ("val_loss", "min"),
    "fusion": ("val_acc", "max"),
}


def _warn_stage_monitor_mismatch(validated: ValidatedConfig, stage: str, label: str) -> None:
    """Log a warning if a stage's monitor/mode diverges from its archetype.

    ``ValidatedConfig`` already enforces checkpoint and early_stopping
    internal agreement. This check is softer: it compares the agreed-upon
    monitor+mode against the expected value for the stage family (e.g.
    fusion must be maximizing ``val_acc``; every other stage must be
    minimizing ``val_loss``). Not fatal — a legitimate override may want a
    different metric — but always interesting to surface.
    """
    expected = _STAGE_MONITORS.get(stage)
    if expected is None:
        return
    exp_monitor, exp_mode = expected
    if validated.checkpoint.monitor != exp_monitor or validated.checkpoint.mode != exp_mode:
        log.warning(
            "stage_monitor_mismatch",
            asset=label,
            stage=stage,
            got=f"{validated.checkpoint.monitor}/{validated.checkpoint.mode}",
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
    # Phase 2: typed view of the rendered jsonnet dict. Assets can read
    # ``resolved.validated.model.class_path`` etc. without re-rendering.
    validated: ValidatedConfig | None = None


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
                OverrideRecord(
                    key="model.init_args.auxiliaries",
                    value="<kd entry>",
                    source="kd",
                )
            )

        # --- Build TLA dict for jsonnet render ---
        # Auto-resume ckpt_path is handled at training time in
        # train_entrypoint.py — NOT here. The orchestrator runs on a
        # different node (NFS-cached); checking exists() here creates a
        # race condition.
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
        )
        if cfg.resource_overrides:
            resources = apply_resource_overrides(resources, cfg.resource_overrides)
            for k, v in cfg.resource_overrides.items():
                audit.append(OverrideRecord(key=k, value=v, source="recipe_resource"))

        # --- Render the jsonnet chain ---
        rendered = render(spec.jsonnet_path, spec.jsonnet_tla)

        # --- Phase 2: Pydantic structural + convention validation ---
        # Raises ConfigValidationError (ValueError subclass) with an
        # actionable message on null list fields, monitor mismatches,
        # un-namespaced class_paths, or unknown top-level keys.
        try:
            validated = validate_config(rendered)
        except ValueError as e:
            raise ValueError(
                f"{cfg.asset_name} config validation (run_dir={paths.run_dir}): {e}"
            ) from e
        _warn_stage_monitor_mismatch(validated, cfg.stage, cfg.asset_name)

        # --- Cross-field validation (Pydantic stage gate) ---
        try:
            validate_stage_config(
                spec=spec,
                resources=resources,
                cfg=cfg,
                merged=rendered,
            )
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
        )

    def resolve_and_validate(
        self,
        cfg: StageConfig,
        *,
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str] | None = None,
    ) -> ResolvedConfig:
        """Thin alias for :meth:`resolve`.

        Kept as a distinct entry point for call-site documentation:
        assets.py / validate.py use this name to signal "I want the full
        validated resolution, not just spec construction". Phase 3
        collapsed the implementation because ``resolve()`` already runs
        ``validate_config`` and the Phase-2-scheduled jsonargparse safety
        net has been deleted along with ``LightningCLI``.
        """
        return self.resolve(
            cfg,
            dataset=dataset,
            seed=seed,
            upstream_ckpts=upstream_ckpts or {},
        )

    # Cross-field validation moved to graphids.config.schemas.
