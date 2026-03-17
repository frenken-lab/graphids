"""Sweep pipeline DAG: 7-step linear pipeline per dataset+scale.

Executes sweeps and full training across all 3 stages (autoencoder → curriculum → fusion),
then evaluates. Each step depends on the previous, with checkpoint dependencies enforced
by the DAG ordering. State is persisted to SQLite for fault-tolerant resume across SLURM restarts.

Usage (via CLI):
    python -m graphids.pipeline.cli sweep-pipeline --dataset set_01 --scale large --num-samples 20
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from graphids.config.constants import (
    PROJECT_ROOT,
    STAGE_MODEL_MAP,
    SWEEP_RESULTS_DIR,
)
from graphids.pipeline.orchestration.job import JobSpec, JobState, ResourceSpec
from graphids.pipeline.orchestration.store import PipelineStore
from graphids.pipeline.subprocess_utils import build_cli_cmd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepStep:
    name: str
    kind: Literal["sweep", "train", "evaluate"]
    stage: str
    model: str


SWEEP_DAG: list[SweepStep] = [
    SweepStep("sweep_autoencoder", "sweep", "autoencoder", STAGE_MODEL_MAP["autoencoder"]),
    SweepStep("train_best_autoencoder", "train", "autoencoder", STAGE_MODEL_MAP["autoencoder"]),
    SweepStep("sweep_curriculum", "sweep", "curriculum", STAGE_MODEL_MAP["curriculum"]),
    SweepStep("train_best_curriculum", "train", "curriculum", STAGE_MODEL_MAP["curriculum"]),
    SweepStep("sweep_fusion", "sweep", "fusion", STAGE_MODEL_MAP["fusion"]),
    SweepStep("train_best_fusion", "train", "fusion", STAGE_MODEL_MAP["fusion"]),
    SweepStep("evaluate", "evaluate", "evaluation", "eval"),
]


# ---------------------------------------------------------------------------
# State management (SQLite via PipelineStore)
# ---------------------------------------------------------------------------

_DB_PATH = PROJECT_ROOT / ".cache" / "kd-gat" / "pipeline.db"


def _sweep_run_id(dataset: str, scale: str) -> str:
    return f"sweep_{dataset}_{scale}"


def _get_store() -> PipelineStore:
    uri = os.getenv("KD_GAT_DB_URI", f"sqlite:///{_DB_PATH}")
    return PipelineStore(uri)


def _ensure_sweep_run(store: PipelineStore, dataset: str, scale: str) -> str:
    """Ensure a sweep run exists in the store, creating jobs for each step."""
    run_id = _sweep_run_id(dataset, scale)
    store.create_run(run_id, {"dataset": dataset, "scale": scale, "kind": "sweep"})

    # Insert jobs for each sweep step if not already present
    jobs = []
    prev_id = None
    for step in SWEEP_DAG:
        job = JobSpec(
            name=f"sweep/{dataset}/{scale}/{step.name}",
            parameters={
                "dataset": dataset,
                "scale": scale,
                "step": step.name,
                "kind": step.kind,
                "stage": step.stage,
                "model": step.model,
            },
            resources=ResourceSpec(gpus=0, cpus=4, memory_gb=16),
            parents=[prev_id] if prev_id else [],
            tags={"pipeline": "sweep"},
        )
        jobs.append(job)
        prev_id = job.id

    store.insert_jobs(run_id, jobs)
    return run_id


def _step_job_id(store: PipelineStore, run_id: str, step_name: str) -> str | None:
    """Find the job ID for a sweep step by name."""
    rows = store.jobs_by_parameter(run_id, "step", step_name)
    return rows[0]["id"] if rows else None


def _step_status(store: PipelineStore, run_id: str, step_name: str) -> str:
    """Get current status of a sweep step."""
    job_id = _step_job_id(store, run_id, step_name)
    if not job_id:
        return "pending"
    return store.current_state(job_id).value


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------


def _sweep_result_path(stage: str, dataset: str, scale: str) -> Path:
    return PROJECT_ROOT / SWEEP_RESULTS_DIR / f"{stage}_{dataset}_{scale}_best.yaml"


def _checkpoint_path(model: str, scale: str, stage: str, dataset: str) -> Path:
    from graphids.config.paths import checkpoint_path
    from graphids.config.resolver import resolve

    cfg = resolve(model, scale, dataset=dataset)
    return checkpoint_path(cfg, stage)


def _metrics_path(scale: str, dataset: str) -> Path:
    from graphids.config.paths import metrics_path
    from graphids.config.resolver import resolve

    cfg = resolve("vgae", scale, dataset=dataset)
    return metrics_path(cfg, "evaluation")


def _verify_step_output(step: SweepStep, dataset: str, scale: str) -> bool:
    if step.kind == "sweep":
        return _sweep_result_path(step.stage, dataset, scale).exists()
    elif step.kind == "train":
        return _checkpoint_path(step.model, scale, step.stage, dataset).exists()
    elif step.kind == "evaluate":
        return _metrics_path(scale, dataset).exists()
    return False


# ---------------------------------------------------------------------------
# Best config loading
# ---------------------------------------------------------------------------


def load_best_config(stage: str, dataset: str, scale: str) -> dict:
    """Read best HP config from a completed sweep's YAML output."""
    import yaml

    path = _sweep_result_path(stage, dataset, scale)
    if not path.exists():
        raise FileNotFoundError(f"No sweep results found at {path}")
    payload = yaml.safe_load(path.read_text())
    return payload["config"]


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------


def _run_sweep_step(
    step: SweepStep,
    dataset: str,
    scale: str,
    *,
    num_samples: int,
    max_concurrent: int,
    tune_epochs: int,
    tune_patience: int,
    warm_start_from: str | None = None,
) -> None:
    """Run a Ray Tune sweep for a single stage."""
    from .tune_config import run_tune

    # Auto-warm-start from set_01 if sweeping a different dataset and results exist
    if warm_start_from is None and dataset != "set_01":
        ref_path = _sweep_result_path(step.stage, "set_01", scale)
        if ref_path.exists():
            warm_start_from = "set_01"
            log.info("Auto warm-starting %s from set_01 results", step.stage)

    log.info(
        "Running sweep: stage=%s, dataset=%s, scale=%s, samples=%d, warm_start_from=%s",
        step.stage,
        dataset,
        scale,
        num_samples,
        warm_start_from,
    )

    run_tune(
        stage=step.stage,
        dataset=dataset,
        scale=scale,
        num_samples=num_samples,
        max_concurrent=max_concurrent,
        max_epochs=tune_epochs,
        patience=tune_patience,
        warm_start_from=warm_start_from,
    )


def _run_train_step(step: SweepStep, dataset: str, scale: str) -> None:
    """Train with best HPs from a completed sweep, using subprocess for CUDA isolation."""
    config = load_best_config(step.stage, dataset, scale)

    cmd = build_cli_cmd(
        step.stage,
        step.model,
        scale,
        dataset,
        overrides=list(config.items()),
        sweep_id=f"tune_{step.stage}_{dataset}_{scale}",
    )
    log.info("Training best %s: %s", step.stage, " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Training step '{step.name}' failed with exit code {result.returncode}")


def _run_evaluate_step(dataset: str, scale: str, *, seed: int | None = None) -> None:
    """Run evaluation stage via subprocess."""
    cmd = build_cli_cmd("evaluation", "vgae", scale, dataset, seeds=str(seed) if seed else None)
    log.info("Running evaluation: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation step failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_sweep_pipeline(
    dataset: str,
    scale: str,
    *,
    num_samples: int = 20,
    max_concurrent: int = 1,
    tune_epochs: int = 50,
    tune_patience: int = 15,
    resume: bool = True,
    dry_run: bool = False,
    multi_seed: bool = False,
) -> None:
    """Execute the 7-step sweep pipeline DAG.

    When multi_seed=True, after the sweep DAG completes, re-trains the best
    config with all DEFAULT_SEEDS for statistical significance reporting.
    """
    if dry_run:
        _dry_run(dataset, scale)
        return

    store = _get_store()
    run_id = _ensure_sweep_run(store, dataset, scale)

    for i, step in enumerate(SWEEP_DAG, 1):
        job_id = _step_job_id(store, run_id, step.name)
        status = store.current_state(job_id).value if job_id else "pending"

        # Skip completed steps with verified output
        if status == "completed" and _verify_step_output(step, dataset, scale):
            log.info("[%d/%d] %s — SKIP (completed)", i, len(SWEEP_DAG), step.name)
            continue

        # Re-run if output is missing despite "completed" status
        if status == "completed" and not _verify_step_output(step, dataset, scale):
            log.warning(
                "[%d/%d] %s — completed but output missing, re-running",
                i,
                len(SWEEP_DAG),
                step.name,
            )

        # Re-run stale "running" steps: check output first
        if status == "running":
            if _verify_step_output(step, dataset, scale):
                log.info(
                    "[%d/%d] %s — was running, output exists, marking completed",
                    i,
                    len(SWEEP_DAG),
                    step.name,
                )
                store.transition(job_id, JobState.COMPLETED, detail="output verified on resume")
                continue
            log.warning(
                "[%d/%d] %s — was running (stale), re-running", i, len(SWEEP_DAG), step.name
            )

        # Execute step
        log.info("[%d/%d] %s — RUNNING", i, len(SWEEP_DAG), step.name)
        attempt_id = store.create_attempt(job_id)
        store.transition(job_id, JobState.RUNNING, attempt_id)

        t0 = time.monotonic()
        try:
            if step.kind == "sweep":
                _run_sweep_step(
                    step,
                    dataset,
                    scale,
                    num_samples=num_samples,
                    max_concurrent=max_concurrent,
                    tune_epochs=tune_epochs,
                    tune_patience=tune_patience,
                )
                # Shutdown Ray between sweeps to prevent state leakage
                try:
                    import ray

                    if ray.is_initialized():
                        ray.shutdown()
                except Exception:
                    pass
            elif step.kind == "train":
                _run_train_step(step, dataset, scale)
            elif step.kind == "evaluate":
                _run_evaluate_step(dataset, scale)

            duration = time.monotonic() - t0
            store.update_attempt(attempt_id, finished_at=datetime.now(UTC).isoformat())
            store.transition(
                job_id,
                JobState.COMPLETED,
                attempt_id,
                detail=f"duration={round(duration, 1)}s",
            )
            log.info("[%d/%d] %s — COMPLETED (%.1fs)", i, len(SWEEP_DAG), step.name, duration)

        except Exception:
            duration = time.monotonic() - t0
            store.update_attempt(attempt_id, finished_at=datetime.now(UTC).isoformat())
            store.transition(
                job_id,
                JobState.FAILED,
                attempt_id,
                detail=f"duration={round(duration, 1)}s",
            )
            log.error("[%d/%d] %s — FAILED after %.1fs", i, len(SWEEP_DAG), step.name, duration)
            raise

    store.close()
    log.info("Sweep pipeline complete for %s/%s", dataset, scale)

    # Multi-seed re-training with best config for statistical significance
    if multi_seed:
        _run_multi_seed_final(dataset, scale)


# ---------------------------------------------------------------------------
# Multi-seed final training
# ---------------------------------------------------------------------------


def _run_multi_seed_final(dataset: str, scale: str) -> None:
    """Re-train best config with all DEFAULT_SEEDS for statistical significance."""
    from graphids.config.constants import DEFAULT_SEEDS

    log.info("=== Multi-seed training for %s/%s with seeds %s ===", dataset, scale, DEFAULT_SEEDS)

    for step in SWEEP_DAG:
        if step.kind != "train":
            continue

        config = load_best_config(step.stage, dataset, scale)
        for seed in DEFAULT_SEEDS:
            cmd = build_cli_cmd(
                step.stage,
                step.model,
                scale,
                dataset,
                seeds=str(seed),
                overrides=list(config.items()),
            )
            log.info("Multi-seed %s (seed=%d): %s", step.stage, seed, " ".join(cmd))
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            if result.returncode != 0:
                log.error("Multi-seed training failed: %s seed=%d", step.stage, seed)
                raise RuntimeError(
                    f"Multi-seed training '{step.stage}' seed={seed} failed "
                    f"with exit code {result.returncode}"
                )

    # Final multi-seed evaluation
    for seed in DEFAULT_SEEDS:
        _run_evaluate_step(dataset, scale, seed=seed)

    log.info("=== Multi-seed training complete for %s/%s ===", dataset, scale)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _dry_run(dataset: str, scale: str) -> None:
    """Print DAG state and verify configs without executing."""
    from graphids.config import data_dir
    from graphids.config.resolver import resolve
    from graphids.pipeline.validate import validate_datasets

    log.info("=== Sweep Pipeline Dry Run: dataset=%s, scale=%s ===", dataset, scale)

    # Verify config resolution + data directory
    ds_errors = validate_datasets([dataset], scale)
    if ds_errors:
        for err in ds_errors:
            log.error("Validation FAILED: %s", err)
        return

    cfg = resolve("vgae", scale, dataset=dataset)
    ddir = data_dir(cfg)
    log.info("Config resolution: OK")
    log.info("Data directory: %s (exists=%s)", ddir, ddir.exists())

    # Load existing state from SQLite store
    store = _get_store()
    run_id = _sweep_run_id(dataset, scale)

    # Print DAG
    log.info("")
    log.info("DAG Steps:")
    for i, step in enumerate(SWEEP_DAG, 1):
        status = _step_status(store, run_id, step.name)
        output_exists = _verify_step_output(step, dataset, scale)
        marker = "OK" if (status == "completed" and output_exists) else status
        log.info(
            "  [%d] %-25s  kind=%-8s stage=%-12s status=%-10s output=%s",
            i,
            step.name,
            step.kind,
            step.stage,
            marker,
            "exists" if output_exists else "missing",
        )

    # Check sweep result paths
    log.info("")
    log.info("Expected outputs:")
    for step in SWEEP_DAG:
        if step.kind == "sweep":
            p = _sweep_result_path(step.stage, dataset, scale)
            log.info("  sweep: %s (exists=%s)", p.relative_to(PROJECT_ROOT), p.exists())
        elif step.kind == "train":
            p = _checkpoint_path(step.model, scale, step.stage, dataset)
            log.info("  train: %s (exists=%s)", p.relative_to(PROJECT_ROOT), p.exists())
        elif step.kind == "evaluate":
            p = _metrics_path(scale, dataset)
            log.info("  eval:  %s (exists=%s)", p.relative_to(PROJECT_ROOT), p.exists())

    store.close()
    log.info("")
    log.info("=== Dry run complete ===")
