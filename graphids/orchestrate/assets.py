"""Dagster asset factory helpers for training.

Each training asset bundles train → test → analyze in a single SLURM job.
Analysis runs inside the GPU job (not in-process on the dagster worker).
"""

from __future__ import annotations

import os

import dagster as dg
from graphids.log import get_logger

from graphids.orchestrate.analysis import build_analysis_spec, supports_analysis
from graphids.orchestrate.execution import slurm_accounting_metadata, touch_complete
from graphids.orchestrate.planning import StageConfig
from graphids.orchestrate.resolve import ConfigResolver
from graphids.slurm import scale_resources

log = get_logger(__name__)


def _runtime_lake_root() -> str:
    return os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns")


def _runtime_user() -> str:
    return os.environ.get("USER", "unknown")


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
        required_resource_keys={"slurm"},
    )
    def _train(context, **upstream_ckpts: str) -> str:
        dataset = context.partition_key.keys_by_dimension["dataset"]
        seed = int(context.partition_key.keys_by_dimension["seed"])
        log.info("asset_start", asset=cfg.asset_name, dataset=dataset,
                 seed=seed, retry=context.retry_number)

        resolver = ConfigResolver(lake_root=_runtime_lake_root(), user=_runtime_user())
        resolved = resolver.resolve(
            cfg, dataset=dataset, seed=seed, upstream_ckpts=upstream_ckpts,
        )

        # Prefer best_model.ckpt, fall back to last.ckpt (fusion RL has no best)
        def _available_ckpt():
            p = resolved.paths
            return p.ckpt_file if p.ckpt_file.exists() else p.last_ckpt_file

        ckpt = _available_ckpt()
        if ckpt.exists() and resolved.paths.complete_marker.exists():
            log.info("asset_skip", asset=cfg.asset_name, ckpt=str(ckpt))
            return str(ckpt)

        resources = resolved.resources
        if context.retry_number > 0:
            original = resources
            for reason in ("OUT_OF_MEMORY", "TIMEOUT"):
                resources = scale_resources(resources, reason)
            log.info("resource_scaled", asset=cfg.asset_name,
                     retry=context.retry_number,
                     old_mem=original.mem, new_mem=resources.mem,
                     old_time=original.time, new_time=resources.time)

        # Build analysis spec if model supports it (runs inside the SLURM job)
        analysis_spec = None
        if has_analysis:
            analysis_spec = build_analysis_spec(
                cfg=cfg, dataset=dataset, seed=seed,
                ckpt_path=str(resolved.paths.ckpt_file),
            )

        def _observe(slurm_state, jid):
            context.log_event(
                dg.AssetObservation(
                    asset_key=context.asset_key,
                    metadata={"slurm_state": slurm_state, "job_id": jid},
                )
            )

        state, job_id = context.resources.slurm.submit_and_wait(
            training_spec=resolved.spec,
            resources=resources,
            job_name=f"{cfg.asset_name}_{dataset}_s{seed}",
            on_state=_observe,
            analysis_spec=analysis_spec,
        )

        if state == "DRY_RUN":
            return str(_available_ckpt())
        if state != "COMPLETED":
            log.error("asset_failed", asset=cfg.asset_name, state=state,
                      job_id=job_id)
            raise RuntimeError(f"SLURM job failed: {state}")

        touch_complete(resolved.paths.run_dir)

        md = {"run_dir": dg.MetadataValue.text(str(resolved.paths.run_dir))}
        wall_time = ""
        if job_id:
            accounting = slurm_accounting_metadata(job_id)
            wall_time = accounting["wall_time"] or ""
            md.update({
                "job_id": dg.MetadataValue.int(accounting["job_id"]),
                "wall_time": dg.MetadataValue.text(wall_time),
                "peak_rss": dg.MetadataValue.text(accounting["peak_rss"] or ""),
            })
        context.add_output_metadata(md)

        log.info("asset_complete", asset=cfg.asset_name, state=state,
                 job_id=job_id, wall_time=wall_time)

        return str(_available_ckpt())

    return _train
