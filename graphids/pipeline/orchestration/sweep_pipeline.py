"""Sweep pipeline DAG: 7-step linear pipeline per dataset+scale.

Executes sweeps and full training across all 3 stages (autoencoder → curriculum → fusion),
then evaluates. Each step depends on the previous, with checkpoint dependencies enforced
by the DAG ordering. State is persisted to JSON for fault-tolerant resume across SLURM restarts.

Usage (via CLI):
    python -m graphids.pipeline.cli sweep-pipeline --dataset set_01 --scale large --num-samples 20
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SweepStep:
    name: str
    kind: Literal["sweep", "train", "evaluate"]
    stage: str
    model: str


SWEEP_DAG: list[SweepStep] = [
    SweepStep("sweep_autoencoder", "sweep", "autoencoder", "vgae"),
    SweepStep("train_best_autoencoder", "train", "autoencoder", "vgae"),
    SweepStep("sweep_curriculum", "sweep", "curriculum", "gat"),
    SweepStep("train_best_curriculum", "train", "curriculum", "gat"),
    SweepStep("sweep_fusion", "sweep", "fusion", "dqn"),
    SweepStep("train_best_fusion", "train", "fusion", "dqn"),
    SweepStep("evaluate", "evaluate", "evaluation", "eval"),
]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

_STATUS = Literal["pending", "running", "completed", "failed"]


def _state_path(dataset: str, scale: str) -> Path:
    return PROJECT_ROOT / "data" / "sweep_state" / f"{dataset}_{scale}_state.json"


def _load_state(dataset: str, scale: str) -> dict[str, Any]:
    path = _state_path(dataset, scale)
    if path.exists():
        return json.loads(path.read_text())
    return {"dataset": dataset, "scale": scale, "steps": {}}


def _save_state(state: dict[str, Any], dataset: str, scale: str) -> None:
    path = _state_path(dataset, scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write via tmp + rename
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _update_step_state(
    state: dict[str, Any],
    step_name: str,
    status: _STATUS,
    dataset: str,
    scale: str,
    **extra: Any,
) -> None:
    if step_name not in state["steps"]:
        state["steps"][step_name] = {}
    state["steps"][step_name]["status"] = status
    state["steps"][step_name].update(extra)
    _save_state(state, dataset, scale)


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------


def _sweep_result_path(stage: str, dataset: str, scale: str) -> Path:
    return PROJECT_ROOT / "data" / "sweep_results" / f"{stage}_{dataset}_{scale}_best.yaml"


def _checkpoint_path(model: str, scale: str, stage: str, dataset: str) -> Path:
    from graphids.config.paths import checkpoint_path_str

    return PROJECT_ROOT / checkpoint_path_str(dataset, model, scale, stage)


def _metrics_path(scale: str, dataset: str) -> Path:
    from graphids.config.paths import metrics_path_str

    return PROJECT_ROOT / metrics_path_str(dataset, "vgae", scale, "evaluation")


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
) -> None:
    """Run a Ray Tune sweep for a single stage."""
    from .tune_config import run_tune

    log.info(
        "Running sweep: stage=%s, dataset=%s, scale=%s, samples=%d",
        step.stage,
        dataset,
        scale,
        num_samples,
    )

    run_tune(
        stage=step.stage,
        dataset=dataset,
        scale=scale,
        num_samples=num_samples,
        max_concurrent=max_concurrent,
        max_epochs=tune_epochs,
        patience=tune_patience,
    )


def _run_train_step(step: SweepStep, dataset: str, scale: str) -> None:
    """Train with best HPs from a completed sweep, using subprocess for CUDA isolation."""
    config = load_best_config(step.stage, dataset, scale)

    cmd = [
        sys.executable,
        "-m",
        "graphids.pipeline.cli",
        step.stage,
        "--model",
        step.model,
        "--scale",
        scale,
        "--dataset",
        dataset,
    ]
    for key, value in config.items():
        cmd.extend(["-O", key, str(value)])

    cmd.extend(["--sweep-id", f"tune_{step.stage}_{dataset}_{scale}"])
    log.info("Training best %s: %s", step.stage, " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Training step '{step.name}' failed with exit code {result.returncode}")


def _run_evaluate_step(dataset: str, scale: str) -> None:
    """Run evaluation stage via subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "graphids.pipeline.cli",
        "evaluation",
        "--model",
        "vgae",
        "--scale",
        scale,
        "--dataset",
        dataset,
    ]
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
) -> None:
    """Execute the 7-step sweep pipeline DAG."""
    if dry_run:
        _dry_run(dataset, scale)
        return

    state = (
        _load_state(dataset, scale) if resume else {"dataset": dataset, "scale": scale, "steps": {}}
    )

    for i, step in enumerate(SWEEP_DAG, 1):
        step_state = state.get("steps", {}).get(step.name, {})
        status = step_state.get("status", "pending")

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
                _update_step_state(
                    state,
                    step.name,
                    "completed",
                    dataset,
                    scale,
                    completed_at=datetime.now(UTC).isoformat(),
                )
                continue
            log.warning(
                "[%d/%d] %s — was running (stale), re-running", i, len(SWEEP_DAG), step.name
            )

        # Execute step
        log.info("[%d/%d] %s — RUNNING", i, len(SWEEP_DAG), step.name)
        started_at = datetime.now(UTC).isoformat()
        _update_step_state(state, step.name, "running", dataset, scale, started_at=started_at)

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
            _update_step_state(
                state,
                step.name,
                "completed",
                dataset,
                scale,
                completed_at=datetime.now(UTC).isoformat(),
                duration_s=round(duration, 1),
            )
            log.info("[%d/%d] %s — COMPLETED (%.1fs)", i, len(SWEEP_DAG), step.name, duration)

        except Exception:
            duration = time.monotonic() - t0
            _update_step_state(
                state,
                step.name,
                "failed",
                dataset,
                scale,
                completed_at=datetime.now(UTC).isoformat(),
                duration_s=round(duration, 1),
            )
            log.error("[%d/%d] %s — FAILED after %.1fs", i, len(SWEEP_DAG), step.name, duration)
            raise

    log.info("Sweep pipeline complete for %s/%s", dataset, scale)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _dry_run(dataset: str, scale: str) -> None:
    """Print DAG state and verify configs without executing."""
    from graphids.config import data_dir
    from graphids.config.resolver import resolve

    log.info("=== Sweep Pipeline Dry Run: dataset=%s, scale=%s ===", dataset, scale)

    # Verify config resolution
    try:
        cfg = resolve("vgae", scale, dataset=dataset)
        ddir = data_dir(cfg)
        log.info("Config resolution: OK")
        log.info("Data directory: %s (exists=%s)", ddir, ddir.exists())
    except Exception as e:
        log.error("Config resolution FAILED: %s", e)
        return

    # Load existing state
    state = _load_state(dataset, scale)

    # Print DAG
    log.info("")
    log.info("DAG Steps:")
    for i, step in enumerate(SWEEP_DAG, 1):
        step_state = state.get("steps", {}).get(step.name, {})
        status = step_state.get("status", "pending")
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

    log.info("")
    log.info("=== Dry run complete ===")
