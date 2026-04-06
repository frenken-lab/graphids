"""Dagster asset factory helpers for training.

Each training asset bundles train → test → analyze in a single SLURM job.
Analysis runs inside the GPU job (not in-process on the dagster worker).
"""

import dagster as dg

from graphids.log import get_logger
from graphids.orchestrate.analysis import build_analysis_spec, supports_analysis
from graphids.orchestrate.dagster.asset_config import (
    TrainingAssetConfig,  # noqa: TC001 — Dagster resolves at runtime
)
from graphids.orchestrate.dagster.runtime import (
    _runtime_lake_root,
    _runtime_user,
    _touch_complete,
    partition_keys,
)
from graphids.orchestrate.dagster.resources import SlurmTrainingResource
from graphids.orchestrate.planning import StageConfig
from graphids.orchestrate.resolve import ConfigResolver
from graphids.slurm import job_accounting, scale_resources

log = get_logger(__name__)


def _asset_description(cfg: StageConfig) -> str:
    """Human-readable asset description for dagster UI and CLI."""
    parts = [cfg.model_type, cfg.scale]
    for k, v in sorted(cfg.model_init_overrides.items()):
        parts.append(f"{k}={v}")
    return f"{cfg.stage} ({', '.join(parts)})"


def make_training_asset(
    cfg: StageConfig,
    partitions_def: dg.MultiPartitionsDefinition,
) -> dg.AssetsDefinition:
    """Build one partitioned training asset that runs train→test→analyze in one SLURM job."""
    ins = {name: dg.AssetIn(key=dg.AssetKey(name)) for name in cfg.upstream_asset_names}
    has_analysis = supports_analysis(cfg.model_type)

    @dg.asset(
        name=cfg.asset_name,
        ins=ins,
        partitions_def=partitions_def,
        retry_policy=dg.RetryPolicy(max_retries=2, delay=30),
        group_name=cfg.stage,
        kinds={"checkpoint"},
        tags={"stage": cfg.stage, "model_type": cfg.model_type, "scale": cfg.scale},
        description=_asset_description(cfg),
    )
    def _train(
        context: dg.AssetExecutionContext,
        config: TrainingAssetConfig,
        slurm: dg.ResourceParam[SlurmTrainingResource],
        **upstream_ckpts: str,
    ) -> dg.Output[str]:
        dataset, seed = partition_keys(context)
        log.info(
            "asset_start",
            asset=cfg.asset_name,
            dataset=dataset,
            seed=seed,
            retry=context.retry_number,
        )

        # resolve_and_validate: pre-SLURM Pydantic + cross-field gate (ADR 0009,
        # jsonargparse schema pass removed in Phase 3 — 2026-04-05).
        resolver = ConfigResolver(lake_root=_runtime_lake_root(), user=_runtime_user())
        resolved = resolver.resolve_and_validate(
            cfg,
            dataset=dataset,
            seed=seed,
            upstream_ckpts=upstream_ckpts,
        )

        ckpt = resolved.paths.resolved_ckpt
        if ckpt.exists() and resolved.paths.complete_marker.exists():
            log.info("asset_skip", asset=cfg.asset_name, ckpt=str(ckpt))
            return dg.Output(str(ckpt), metadata={"skipped": dg.MetadataValue.bool(True)})

        resources = resolved.resources
        if context.retry_number > 0:
            original = resources
            for reason in ("OUT_OF_MEMORY", "TIMEOUT"):
                resources = scale_resources(resources, reason)
            log.info(
                "resource_scaled",
                asset=cfg.asset_name,
                retry=context.retry_number,
                old_mem=original.mem,
                new_mem=resources.mem,
                old_time=original.time,
                new_time=resources.time,
            )

        # Build analysis spec if model supports it (runs inside the SLURM job)
        analysis_spec = None
        if has_analysis and config.run_analysis:
            analysis_spec = build_analysis_spec(
                cfg=cfg,
                dataset=dataset,
                seed=seed,
                ckpt_path=str(resolved.paths.ckpt_file),
            )

        def _observe(slurm_state, jid):
            context.log_event(
                dg.AssetObservation(
                    asset_key=context.asset_key,
                    metadata={"slurm_state": slurm_state, "job_id": jid},
                )
            )

        state, job_id = slurm.submit_and_wait(
            training_spec=resolved.spec,
            resources=resources,
            job_name=f"{cfg.asset_name}_{dataset}_s{seed}",
            on_state=_observe,
            run_test=config.run_test,
            analysis_spec=analysis_spec,
            dry_run=config.dry_run,
        )

        if state == "DRY_RUN":
            return dg.Output(str(resolved.paths.resolved_ckpt), metadata={})
        if state != "COMPLETED":
            log.error("asset_failed", asset=cfg.asset_name, state=state, job_id=job_id)
            raise RuntimeError(f"SLURM job failed: {state}")

        _touch_complete(resolved.paths.run_dir)

        md = {"run_dir": dg.MetadataValue.text(str(resolved.paths.run_dir))}
        wall_time = ""
        if job_id:
            accounting = job_accounting(job_id)
            wall_time = accounting["wall_time"] or ""
            md.update(
                {
                    "job_id": dg.MetadataValue.int(accounting["job_id"]),
                    "wall_time": dg.MetadataValue.text(wall_time),
                    "peak_rss": dg.MetadataValue.text(accounting["peak_rss"] or ""),
                }
            )

        # Enrich from sidecar if available
        record_path = resolved.paths.run_dir / "run_record.json"
        if record_path.exists():
            from graphids.core.run_record import RunRecord

            try:
                record = RunRecord.model_validate_json(record_path.read_text())
                for k, v in record.metrics.items():
                    md[f"metric_{k}"] = dg.MetadataValue.float(v)
                if record.wall_time_seconds is not None:
                    md["wall_time_seconds"] = dg.MetadataValue.float(record.wall_time_seconds)
                for phase, ok in record.phases.items():
                    md[f"phase_{phase}"] = dg.MetadataValue.bool(ok)
            except Exception as e:
                log.warning("sidecar_read_failed", path=str(record_path), error=str(e))

        log.info(
            "asset_complete", asset=cfg.asset_name, state=state, job_id=job_id, wall_time=wall_time
        )

        return dg.Output(str(resolved.paths.resolved_ckpt), metadata=md)

    return _train
