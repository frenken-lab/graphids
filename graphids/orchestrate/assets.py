"""Dagster asset factory helpers for training and analysis."""

from __future__ import annotations

from typing import Any, Protocol

import dagster as dg

from graphids.core.analyze_entrypoint import run_analysis_from_spec
from graphids.orchestrate.analysis import build_analysis_spec, output_status, write_manifest
from graphids.orchestrate.execution import artifact_paths, slurm_accounting_metadata, touch_complete, training_spec
from graphids.orchestrate.planning import StageConfig
from graphids.slurm import apply_resource_overrides, get_resources, scale_resources


class TrainingSubmitter(Protocol):
    """Protocol for resources capable of submitting and waiting on training jobs."""

    def submit_and_wait(
        self,
        training_spec,
        resources,
        job_name: str,
        on_state=None,
    ) -> tuple[str, int]:
        ...


def make_training_asset(
    cfg: StageConfig,
    partitions_def: dg.MultiPartitionsDefinition,
    lake_root: str,
    user: str,
) -> dg.AssetsDefinition:
    """Build one partitioned training asset from stage config."""
    ins = {name: dg.AssetIn(key=dg.AssetKey(name)) for name in cfg.upstream_asset_names}
    is_eval = cfg.stage == "evaluation"

    @dg.asset(
        name=cfg.asset_name,
        ins=ins,
        partitions_def=partitions_def,
        retry_policy=dg.RetryPolicy(max_retries=2, delay=30),
        group_name=cfg.stage,
        kinds={"metrics"} if is_eval else {"checkpoint"},
        tags={"stage": cfg.stage, "model_type": cfg.model_type, "scale": cfg.scale},
        description=f"{cfg.stage} ({cfg.model_type}, {cfg.scale})",
        required_resource_keys={"slurm"},
    )
    def _train(context, **upstream_ckpts: str) -> str:
        dataset = context.partition_key.keys_by_dimension["dataset"]
        seed = int(context.partition_key.keys_by_dimension["seed"])

        rd, rd_path, ckpt_file, complete = artifact_paths(
            cfg,
            lake_root=lake_root,
            user=user,
            dataset=dataset,
            seed=seed,
        )

        if ckpt_file.exists() and complete.exists():
            context.log.info(f"Already complete: {ckpt_file}")
            return str(ckpt_file)

        spec = training_spec(
            cfg,
            dataset=dataset,
            seed=seed,
            run_directory=rd,
            run_directory_path=rd_path,
            upstream_ckpts=upstream_ckpts,
        )

        resources = get_resources(cfg.resource_model or cfg.model_type, cfg.scale, cfg.stage)
        if cfg.resource_overrides:
            resources = apply_resource_overrides(resources, cfg.resource_overrides)
        if context.retry_number > 0:
            for reason in ("OUT_OF_MEMORY", "TIMEOUT"):
                resources = scale_resources(resources, reason)

        def _observe(slurm_state, jid):
            context.log_event(
                dg.AssetObservation(
                    asset_key=context.asset_key,
                    metadata={"slurm_state": slurm_state, "job_id": jid},
                )
            )

        state, job_id = context.resources.slurm.submit_and_wait(
            training_spec=spec,
            resources=resources,
            job_name=f"{cfg.asset_name}_{dataset}_s{seed}",
            on_state=_observe,
        )

        if state == "DRY_RUN":
            return str(ckpt_file)
        if state != "COMPLETED":
            raise RuntimeError(f"SLURM job failed: {state}")

        touch_complete(rd_path)

        if job_id:
            accounting = slurm_accounting_metadata(job_id)
            context.add_output_metadata(
                {
                    "job_id": dg.MetadataValue.int(accounting["job_id"]),
                    "wall_time": dg.MetadataValue.text(accounting["wall_time"] or ""),
                    "peak_rss": dg.MetadataValue.text(accounting["peak_rss"] or ""),
                }
            )

        return str(ckpt_file)

    return _train


def make_analysis_asset(
    cfg: StageConfig,
    partitions_def: dg.MultiPartitionsDefinition,
) -> dg.AssetsDefinition:
    """Build analysis asset that consumes a training checkpoint."""

    @dg.asset(
        name=f"{cfg.asset_name}_analysis",
        ins={"checkpoint_path": dg.AssetIn(key=dg.AssetKey(cfg.asset_name))},
        partitions_def=partitions_def,
        group_name="analysis",
        kinds={"artifact", "report"},
        tags={"stage": cfg.stage, "model_type": cfg.model_type, "scale": cfg.scale},
        description=f"Analysis artifacts for {cfg.asset_name}",
    )
    def _analyze(context, checkpoint_path: str) -> str:
        dataset = context.partition_key.keys_by_dimension["dataset"]
        seed = int(context.partition_key.keys_by_dimension["seed"])
        spec = build_analysis_spec(
            cfg=cfg,
            dataset=dataset,
            seed=seed,
            ckpt_path=checkpoint_path,
        )

        run_analysis_from_spec(spec)
        manifest_path = write_manifest(
            asset_name=cfg.asset_name,
            dataset=dataset,
            seed=seed,
            checkpoint_path=checkpoint_path,
            spec=spec,
        )
        expected_outputs, existing_outputs = output_status(spec)

        context.add_output_metadata(
            {
                "manifest": dg.MetadataValue.path(str(manifest_path)),
                "output_dir": dg.MetadataValue.path(spec.output_dir),
                "expected_outputs": len(expected_outputs),
                "existing_outputs": len(existing_outputs),
            }
        )
        return str(manifest_path)

    return _analyze
