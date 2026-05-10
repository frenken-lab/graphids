"""Execution helpers for the new experiment seam.

Ray/Hydra can attach here later. For now this module gives us a single place to
write manifests and events around any callable run body.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from graphids._mlflow import make_logger
from graphids.exp.config import AnalyzeConfig, ExtractConfig, RunConfig, RunSummary
from graphids.exp.journal import (
    EventRecord,
    append_event,
    load_events,
    load_manifest,
    write_manifest,
)


def _payload(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": repr(obj)}


def run_stage(run: RunConfig) -> dict[str, Any] | None:
    """Default stage dispatcher for experiment launches.

    Fit/test are not wired yet in the new primitives layer; extract/analyze
    can already run off the new config objects.
    """
    if run.stage in {"fit", "test"}:
        raise NotImplementedError(
            f"stage {run.stage!r} is not wired to the new primitive runner yet"
        )
    if run.stage == "extract":
        from graphids.core.data.extract import extract_states

        spec = ExtractConfig.model_validate(
            {
                **run.config,
                "name": run.name,
                "action": "extract",
                "plan_id": run.plan_id or run.name,
                "resources": run.resources.model_dump(mode="json"),
            }
        )
        extract_states(
            checkpoints=spec.extractor_ckpts,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            max_samples=spec.max_samples,
            max_val_samples=spec.max_val_samples,
            batch_size=spec.batch_size,
            seed=spec.seed,
            val_fraction=spec.val_fraction,
            representation_cfg=spec.representation_cfg,
        )
        return {"stage": "extract", "output_dir": spec.output_dir}
    if run.stage == "analyze":
        from graphids.core.artifacts.analyzer import Analyzer

        spec = AnalyzeConfig.model_validate(
            {
                **run.config,
                "name": run.name,
                "action": "analyze",
                "plan_id": run.plan_id or run.name,
                "resources": run.resources.model_dump(mode="json"),
            }
        )
        Analyzer(spec).run()
        return {"stage": "analyze", "output_dir": spec.output_dir}
    if run.stage in {"cache", "hf_push"}:
        raise NotImplementedError(f"stage {run.stage!r} is not wired yet")
    raise ValueError(f"unknown stage: {run.stage!r}")


def launch_run(
    run: RunConfig,
) -> RunSummary:
    """Run one launchable config with manifest/event tracking."""
    backend = run.resources.backend
    if backend == "ray":
        try:
            import ray  # noqa: F401
        except ImportError:
            backend = "local"
    logger = make_logger(
        experiment_name=f"graphids/{run.dataset or 'unknown'}/{run.stage}",
        run_name=run.name,
        tags=run.mlflow_tags(),
        artifact_location=str(run.outputs.mlflow_dir()),
    )
    logger.log_hyperparams(run.mlflow_hparams(backend=backend))
    manifest = run.journal_manifest(status="running")
    write_manifest(run.outputs.run_dir, manifest, name=run.outputs.manifest_name)
    append_event(
        run.outputs.run_dir,
        EventRecord(
            status="running",
            stage=run.stage,
            message="launch_started",
            details={"backend": backend},
        ),
        name=run.outputs.events_name,
    )

    try:
        if backend == "ray":
            import ray

            ray.init(ignore_reinit_error=True, include_dashboard=False)
            result = ray.get(ray.remote(run_stage).remote(run))
        else:
            result = run_stage(run)
        if logger.run_id is not None:
            logger.experiment.set_terminated(logger.run_id, status="FINISHED")
        append_event(
            run.outputs.run_dir,
            EventRecord(status="finished", stage=run.stage, message="run_finished", details=_payload(result)),
            name=run.outputs.events_name,
        )
        write_manifest(
            run.outputs.run_dir,
            manifest.model_copy(update={"status": "finished"}),
            name=run.outputs.manifest_name,
        )
        return RunSummary(
            run_dir=str(run.outputs.run_dir),
            status="finished",
            stage=run.stage,
            name=run.name,
            last_event="run_finished",
        )
    except BaseException as exc:  # noqa: BLE001 - record all failures, then re-raise
        failure = f"{type(exc).__name__}: {exc}"
        if logger.run_id is not None:
            logger.experiment.set_terminated(logger.run_id, status="FAILED")
        append_event(
            run.outputs.run_dir,
            EventRecord(
                status="failed",
                stage=run.stage,
                message="run_failed",
                details={"failure": failure},
            ),
            name=run.outputs.events_name,
        )
        write_manifest(
            run.outputs.run_dir,
            manifest.model_copy(update={"status": "failed", "failure": failure}),
            name=run.outputs.manifest_name,
        )
        raise


def summarize_run(run_dir: str | Path) -> RunSummary | None:
    manifest = load_manifest(run_dir)
    if manifest is None:
        return None
    events = load_events(run_dir)
    last = events[-1].message if events else None
    err = manifest.failure
    return RunSummary(
        run_dir=manifest.run_dir,
        status=manifest.status,
        stage=manifest.stage,
        name=manifest.name,
        last_event=last,
        error=err,
        extra={"git_sha": manifest.git_sha, "run_id": manifest.run_id},
    )
