"""Stateful SLURM job coordinator with reactive failure handling.

Submits pipeline stages as individual SLURM jobs (GPU stages to gpu partition,
CPU stages to cpu partition), polls sacct for completion, reacts to failures
with adjusted resources, and verifies artifacts after each stage completes.

Usage (via cli.py):
    python -m graphids.pipeline.cli coordinator --dataset hcrl_sa --seeds 42,123
    python -m graphids.pipeline.cli coordinator --resume-state .cache/kd-gat/pipeline_state.json
    python -m graphids.pipeline.cli coordinator --dataset hcrl_sa --seeds 42 --dry-run
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphids.config.constants import (
    PROJECT_ROOT,
    SLURM_ACCOUNT,
    SLURM_GPU_TYPE,
    STAGE_DEPENDENCIES,
)

from .state import StageStatus, load_state, now_iso, save_state, update_stage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource profiles per stage type
# ---------------------------------------------------------------------------

# Base profiles keyed by partition type. Stages look up by
# {model}_{scale}_{stage}, then {model}_{stage}, then {stage}, then default.
RESOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "preprocess": {"partition": "cpu", "gpu": 0, "cpus": 4, "mem": "32G", "time": "1:00:00"},
    # VGAE
    "vgae_large_autoencoder": {
        "partition": "gpu",
        "gpu": 1,
        "cpus": 4,
        "mem": "20G",
        "time": "3:00:00",
    },
    "vgae_small_autoencoder": {
        "partition": "gpu",
        "gpu": 1,
        "cpus": 4,
        "mem": "16G",
        "time": "2:00:00",
    },
    # GAT (curriculum and normal share the same profile per scale)
    "gat_large_curriculum": {
        "partition": "gpu",
        "gpu": 1,
        "cpus": 4,
        "mem": "16G",
        "time": "3:00:00",
    },
    "gat_large_normal": {"partition": "gpu", "gpu": 1, "cpus": 4, "mem": "16G", "time": "3:00:00"},
    "gat_small_curriculum": {
        "partition": "gpu",
        "gpu": 1,
        "cpus": 4,
        "mem": "12G",
        "time": "1:30:00",
    },
    "gat_small_normal": {"partition": "gpu", "gpu": 1, "cpus": 4, "mem": "12G", "time": "1:30:00"},
    # DQN fusion + evaluation are CPU-only
    "dqn_fusion": {"partition": "cpu", "gpu": 0, "cpus": 4, "mem": "16G", "time": "0:30:00"},
    "evaluation": {"partition": "cpu", "gpu": 0, "cpus": 4, "mem": "16G", "time": "0:30:00"},
    "aggregate": {"partition": "cpu", "gpu": 0, "cpus": 2, "mem": "8G", "time": "0:15:00"},
}

_DEFAULT_RESOURCE = {"partition": "gpu", "gpu": 1, "cpus": 4, "mem": "20G", "time": "3:00:00"}

# ---------------------------------------------------------------------------
# Failure reaction table
# ---------------------------------------------------------------------------

FAILURE_REACTIONS: dict[str, dict[str, Any]] = {
    "OUT_OF_MEMORY": {"action": "retry", "adjust_mem": 2.0, "max_retries": 2},
    "TIMEOUT": {"action": "retry", "adjust_time": 1.5, "max_retries": 2},
    "NODE_FAIL": {"action": "retry", "exclude_node": True, "max_retries": 3},
    "CANCELLED": {"action": "pause"},
    "FAILED": {"action": "retry", "max_retries": 1},
    "MISSING_ARTIFACTS": {"action": "retry", "max_retries": 1},
}

# Backoff: base delay (seconds) and multiplier per attempt
_RETRY_BACKOFF_BASE = 60
_RETRY_BACKOFF_FACTOR = 2.0
_RETRY_BACKOFF_MAX = 600  # cap at 10 minutes


# ---------------------------------------------------------------------------
# Helpers (pure functions, no class state)
# ---------------------------------------------------------------------------


def _scale_mem(mem_str: str, factor: float) -> str:
    """Scale a memory string like '16G' by a factor."""
    value = int(mem_str.rstrip("GgMm"))
    unit = mem_str[-1].upper()
    return f"{int(value * factor)}{unit}"


def _scale_time(time_str: str, factor: float) -> str:
    """Scale a time string like '3:00:00' by a factor."""
    parts = time_str.split(":")
    if len(parts) == 3:
        total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        total_seconds = int(parts[0]) * 60 + int(parts[1])
    else:
        total_seconds = int(parts[0])

    total_seconds = int(total_seconds * factor)
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _time_to_hours(time_str: str) -> float:
    """Convert 'H:MM:SS' to fractional hours."""
    parts = time_str.split(":")
    return int(parts[0]) + int(parts[1]) / 60 if len(parts) >= 2 else 0.0


def _sacct_query(job_id: int, fmt: str) -> str | None:
    """Query sacct for a single field. Returns first non-empty line or None."""
    result = subprocess.run(
        ["sacct", "-j", str(job_id), f"--format={fmt}", "--noheader", "-P", "--parsable2"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        log.warning("sacct failed for job %d: %s", job_id, result.stderr.strip())
        return None
    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    return lines[0] if lines else None


def _sacct_resources(job_id: int) -> dict[str, str]:
    """Query sacct for actual resource usage. Returns dict of field→value."""
    fmt = "MaxRSS,Elapsed,MaxVMSize,TotalCPU"
    result = subprocess.run(
        ["sacct", "-j", str(job_id), f"--format={fmt}", "--noheader", "-P", "--parsable2"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return {}
    fields = ["max_rss", "elapsed", "max_vmsize", "total_cpu"]
    for line in result.stdout.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) == len(fields) and parts[0]:  # skip batch step with empty values
            return dict(zip(fields, parts))
    return {}


def _find_checkpoint(experiment_root: str, run_id_str: str) -> str | None:
    """Find the most recent Lightning auto-checkpoint in persistent experiment dir.

    Lightning's SLURMEnvironment saves to {default_root_dir}/.pl_auto_save.ckpt
    on SIGUSR1. We also check for any .ckpt files.
    """
    run_dir = Path(experiment_root) / run_id_str
    if not run_dir.exists():
        return None

    # Lightning's standard auto-save filename
    auto_save = run_dir / ".pl_auto_save.ckpt"
    if auto_save.exists():
        return str(auto_save)

    # Fallback: any .ckpt file (sorted by mtime, newest first)
    ckpts = sorted(run_dir.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(ckpts[0]) if ckpts else None


def _retry_delay(attempt: int) -> float:
    """Exponential backoff delay for retry attempt (1-indexed)."""
    delay = _RETRY_BACKOFF_BASE * (_RETRY_BACKOFF_FACTOR ** (attempt - 1))
    return min(delay, _RETRY_BACKOFF_MAX)


def _parse_sacct_mem(mem_str: str) -> float:
    """Parse sacct memory string (e.g. '12345K', '5678M', '2G') to GB."""
    if not mem_str or not mem_str[0].isdigit():
        return 0.0
    unit = mem_str[-1].upper()
    value = float(mem_str[:-1]) if unit in "KMG" else float(mem_str)
    if unit == "K":
        return value / (1024**2)
    if unit == "M":
        return value / 1024
    if unit == "G":
        return value
    return value / (1024**3)  # assume bytes


def _parse_elapsed(elapsed: str) -> float:
    """Parse sacct elapsed time (e.g. '1:23:45', '0:05:30') to hours."""
    parts = elapsed.split(":")
    if len(parts) == 3:
        return int(parts[0]) + int(parts[1]) / 60 + int(parts[2]) / 3600
    if len(parts) == 2:
        return int(parts[0]) / 60 + int(parts[1]) / 3600
    return 0.0


def _round_up_mem(gb: float) -> str:
    """Round memory up to next integer GB (minimum 4G)."""
    return f"{max(4, int(gb) + (1 if gb % 1 > 0 else 0))}G"


def _round_up_time(hours: float) -> str:
    """Round time up to next 30-minute block (minimum 0:30:00)."""
    half_hours = max(1, int(hours * 2) + (1 if hours * 2 % 1 > 0 else 0))
    h, half = divmod(half_hours, 2)
    return f"{h}:{30 if half else '00'}:00"


_PROFILE_PATH = PROJECT_ROOT / ".cache" / "kd-gat" / "resource_profile.jsonl"
_MIN_SAMPLES = 2  # need at least 2 data points to override defaults
_SAFETY_MARGIN = 1.25  # 25% headroom on top of p95


def _load_resource_history() -> dict[str, list[dict]]:
    """Load historical resource profiles grouped by stage_type+scale key."""
    if not _PROFILE_PATH.exists():
        return {}
    groups: dict[str, list[dict]] = {}
    for line in _PROFILE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = f"{entry['stage_type']}_{entry.get('scale', 'large')}"
        # Normalize: stage_type is "vgae_autoencoder", key becomes "vgae_autoencoder_large"
        # But RESOURCE_PROFILES uses "vgae_large_autoencoder". Remap.
        parts = entry["stage_type"].split("_", 1)
        if len(parts) == 2:
            key = f"{parts[0]}_{entry.get('scale', 'large')}_{parts[1]}"
        else:
            key = entry["stage_type"]
        groups.setdefault(key, []).append(entry.get("actual", {}))
    return groups


def _percentile(values: list[float], pct: float) -> float:
    """Simple percentile calculation (no numpy dependency)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def apply_resource_history() -> int:
    """Read historical profiles and patch RESOURCE_PROFILES with right-sized values.

    Only overrides mem and time; preserves partition, gpu, cpus.
    Returns count of profiles updated.
    """
    history = _load_resource_history()
    if not history:
        return 0

    updated = 0
    for key, samples in history.items():
        if len(samples) < _MIN_SAMPLES:
            continue
        if key not in RESOURCE_PROFILES:
            continue

        # Parse memory values (MaxRSS)
        mem_values = [_parse_sacct_mem(s.get("max_rss", "")) for s in samples]
        mem_values = [v for v in mem_values if v > 0]

        # Parse elapsed times
        time_values = [_parse_elapsed(s.get("elapsed", "")) for s in samples]
        time_values = [v for v in time_values if v > 0]

        profile = RESOURCE_PROFILES[key]
        changed = False

        if len(mem_values) >= _MIN_SAMPLES:
            p95_mem = _percentile(mem_values, 95) * _SAFETY_MARGIN
            new_mem = _round_up_mem(p95_mem)
            old_mem_gb = int(profile["mem"].rstrip("GgMm"))
            new_mem_gb = int(new_mem.rstrip("G"))
            # Only adjust if meaningfully different (>2G change)
            if abs(new_mem_gb - old_mem_gb) > 2:
                profile["mem"] = new_mem
                changed = True

        if len(time_values) >= _MIN_SAMPLES:
            p95_time = _percentile(time_values, 95) * _SAFETY_MARGIN
            new_time = _round_up_time(p95_time)
            # Only adjust if new time differs from current
            if new_time != profile["time"]:
                profile["time"] = new_time
                changed = True

        if changed:
            updated += 1
            log.info(
                "Resource profile %s adjusted from history (%d samples): mem=%s, time=%s",
                key,
                len(samples),
                profile["mem"],
                profile["time"],
            )

    return updated


# ---------------------------------------------------------------------------
# Stage plan builder
# ---------------------------------------------------------------------------


def _get_resources(model_type: str, scale: str, stage: str) -> dict[str, Any]:
    """Get resource profile for a stage, falling back through key variants."""
    for key in (f"{model_type}_{scale}_{stage}", f"{model_type}_{stage}", stage):
        if key in RESOURCE_PROFILES:
            return dict(RESOURCE_PROFILES[key])
    return dict(_DEFAULT_RESOURCE)


def build_stage_plan(
    datasets: list[str],
    seeds: list[int],
    scale: str = "large",
    auxiliaries: str = "none",
) -> dict[str, dict[str, Any]]:
    """Build the complete stage plan with dependencies and resources.

    Each key is: {dataset}/{model}_{scale}_{stage}/seed_{seed}

    The plan covers the standard pipeline:
      autoencoder (VGAE) → curriculum (GAT) → fusion (DQN) → evaluation
    """
    plan: dict[str, dict[str, Any]] = {}

    pipeline_stages = [
        ("vgae", "autoencoder"),
        ("gat", "curriculum"),
        ("dqn", "fusion"),
    ]

    def _make_key(ds: str, model: str, stg: str, sd: int) -> str:
        return f"{ds}/{model}_{scale}_{stg}/seed_{sd}"

    def _deps_for(ds: str, sd: int, stg: str) -> list[str]:
        return [
            _make_key(ds, dep_model, dep_stage, sd)
            for dep_model, dep_stage in STAGE_DEPENDENCIES.get(stg, [])
        ]

    def _add_stage(key: str, stg: str, model: str, ds: str, sd: int, max_ret: int = 2) -> None:
        plan[key] = {
            "status": "pending",
            "depends_on": _deps_for(ds, sd, stg),
            "resources": _get_resources(model, scale, stg),
            "cli_args": {
                "stage": stg,
                "model": model,
                "scale": scale,
                "dataset": ds,
                "seed": sd,
                "auxiliaries": auxiliaries,
            },
            "attempts": 0,
            "max_retries": max_ret,
        }

    for dataset in datasets:
        for seed in seeds:
            for model_type, stage in pipeline_stages:
                _add_stage(
                    _make_key(dataset, model_type, stage, seed), stage, model_type, dataset, seed
                )

            # Evaluation depends on all three training stages
            eval_key = _make_key(dataset, "eval", "evaluation", seed)
            _add_stage(eval_key, "evaluation", "vgae", dataset, seed, max_ret=1)
            # Override deps: evaluation needs all three, not just STAGE_DEPENDENCIES
            plan[eval_key]["depends_on"] = [
                _make_key(dataset, m, s, seed) for m, s in pipeline_stages
            ]

    return plan


# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------


def _sbatch(stage_key: str, stage_info: dict[str, Any]) -> int:
    """Submit a stage as a SLURM job. Returns the job ID."""
    from graphids.pipeline.subprocess_utils import build_cli_cmd

    res = stage_info["resources"]
    cli = stage_info["cli_args"]

    cli_cmd = build_cli_cmd(
        cli["stage"],
        cli["model"],
        cli["scale"],
        cli["dataset"],
        seed=cli.get("seed"),
        auxiliaries=cli.get("auxiliaries", "none"),
        ckpt_path=cli.get("ckpt_path"),
    )
    # Background Python so _preamble.sh's SIGUSR1 trap can fire
    py_cmd = " ".join(str(p) for p in cli_cmd)
    wrap_cmd = (
        f"source scripts/slurm/_preamble.sh && {py_cmd} & _KD_CHILD_PID=$!; wait $_KD_CHILD_PID"
    )

    safe_key = stage_key.replace("/", "_")
    cmd = [
        "sbatch",
        "--parsable",
        f"--account={SLURM_ACCOUNT}",
        f"--partition={res['partition']}",
        f"--mem={res['mem']}",
        f"--time={res['time']}",
        f"--cpus-per-task={res.get('cpus', 4)}",
        f"--job-name=kd-{safe_key[:30]}",
        f"--output=slurm_logs/%j-{safe_key}.out",
        f"--error=slurm_logs/%j-{safe_key}.err",
        "--signal=B:USR1@180",
    ]
    if res.get("gpu"):
        cmd.append(f"--gres=gpu:{SLURM_GPU_TYPE}:{res['gpu']}")
    exclude = stage_info.get("exclude_nodes")
    if exclude:
        cmd.append(f"--exclude={','.join(exclude)}")
    cmd.extend(["--wrap", wrap_cmd])

    log.info("Submitting: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

    job_id = int(result.stdout.strip().split(";")[0])
    log.info("Submitted %s → job %d", stage_key, job_id)
    return job_id


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class PipelineCoordinator:
    """Stateful SLURM job coordinator with reactive failure handling.

    Runs as a long-lived process (in tmux or a small CPU job). Submits
    pipeline stages as SLURM sub-jobs, polls for completion, reacts to
    failures, and verifies artifacts. State is persisted to JSON for
    resume after coordinator restart.
    """

    def __init__(
        self,
        datasets: list[str],
        seeds: list[int],
        scale: str = "large",
        auxiliaries: str = "none",
        state_path: Path | None = None,
        poll_interval: int = 30,
        dry_run: bool = False,
        exit_hooks: list[list[str]] | None = None,
    ):
        self.datasets = datasets
        self.seeds = seeds
        self.scale = scale
        self.auxiliaries = auxiliaries
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.exit_hooks = exit_hooks or []

        self.state_path = state_path or (PROJECT_ROOT / ".cache" / "kd-gat" / "pipeline_state.json")

        # Right-size resource profiles from historical data (before building plan)
        n_adjusted = apply_resource_history()
        if n_adjusted:
            log.info("Adjusted %d resource profiles from historical runs", n_adjusted)

        # Load or create state
        existing = load_state(self.state_path)
        if existing and "stages" in existing:
            log.info(
                "Resuming from %s (%d stages tracked)", self.state_path, len(existing["stages"])
            )
            self.state = existing
            # Reconcile: add any new stages not already in state
            plan = build_stage_plan(datasets, seeds, scale, auxiliaries)
            for key, info in plan.items():
                if key not in self.state["stages"]:
                    self.state["stages"][key] = info
        else:
            self.state = {
                "pipeline_id": f"{'_'.join(datasets)}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                "created": now_iso(),
                "datasets": datasets,
                "seeds": seeds,
                "scale": scale,
                "auxiliaries": auxiliaries,
                "stages": build_stage_plan(datasets, seeds, scale, auxiliaries),
            }
            save_state(self.state, self.state_path)
            log.info("Created new pipeline state: %d stages", len(self.stages))

    @property
    def stages(self) -> dict[str, dict[str, Any]]:
        return self.state["stages"]

    # ----- Status queries -----

    def _stages_with_status(self, *statuses: StageStatus):
        for key, info in self.stages.items():
            if info.get("status") in statuses:
                yield key, info

    def _all_terminal(self) -> bool:
        terminal = {"completed", "abandoned", "paused"}
        return all(s.get("status") in terminal for s in self.stages.values())

    def _dependencies_met(self, stage_info: dict) -> bool:
        return all(
            self.stages.get(dep, {}).get("status") == "completed"
            for dep in stage_info.get("depends_on", [])
        )

    # ----- Control loop -----

    def run(self) -> None:
        """Main control loop. Runs until all stages complete or are abandoned."""
        self._validate_upfront()

        if self.dry_run:
            self._print_plan()
            return

        log.info("Starting coordinator control loop (poll every %ds)", self.poll_interval)
        iteration = 0

        while not self._all_terminal():
            iteration += 1
            log.debug("--- Iteration %d ---", iteration)

            n_polled = self._poll_running_jobs()
            n_verify_failed = self._verify_completions()
            n_dispatched = self._submit_ready_stages()
            n_retried = self._handle_failures()

            save_state(self.state, self.state_path)

            if iteration % 10 == 1 or n_polled or n_dispatched or n_retried:
                self._log_summary()

            if self._detect_deadlock():
                break

            time.sleep(self.poll_interval)

        # --- Exit: summary + resource profiling + hooks ---
        self._log_summary()
        self._save_resource_profile()
        abandoned = list(self._stages_with_status("abandoned"))
        if abandoned:
            log.error("Pipeline finished with %d abandoned stages:", len(abandoned))
            for key, info in abandoned:
                log.error(
                    "  %s: %s", key, info.get("reason", info.get("failure_reason", "unknown"))
                )
        else:
            log.info("Pipeline complete — all stages succeeded.")

        self._run_exit_hooks(success=len(abandoned) == 0)

    # ----- Polling -----

    def _poll_running_jobs(self) -> int:
        """Check sacct for all submitted/running stages. Returns count of state changes."""
        changes = 0
        for key, info in self._stages_with_status("submitted", "running"):
            job_id = info.get("slurm_job_id")
            if not job_id:
                continue

            slurm_state = _sacct_query(job_id, "State")
            if slurm_state is None:
                continue

            if slurm_state == "RUNNING" and info["status"] == "submitted":
                info["status"] = "running"
                info["started"] = now_iso()
                changes += 1
                log.info("Stage %s now RUNNING (job %d)", key, job_id)

            elif slurm_state == "COMPLETED":
                info["status"] = "completed"
                info["completed"] = now_iso()
                # Phase 4: Capture actual resource usage for profiling
                actual = _sacct_resources(job_id)
                if actual:
                    info["actual_resources"] = actual
                changes += 1
                log.info("Stage %s COMPLETED (job %d)", key, job_id)

            elif slurm_state in ("FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "CANCELLED"):
                info["status"] = "failed"
                info["failure_reason"] = slurm_state
                info["completed"] = now_iso()
                changes += 1
                # Capture node for potential exclusion
                node = _sacct_query(job_id, "NodeList")
                if node and node != "None assigned":
                    info["failed_node"] = node
                log.warning("Stage %s FAILED: %s (job %d)", key, slurm_state, job_id)

            # PENDING = still in queue, no action needed

        return changes

    # ----- Submission -----

    def _submit_ready_stages(self) -> int:
        """Submit stages whose dependencies are met. Returns count submitted."""
        submitted = 0
        for key, info in self._stages_with_status("pending", "retry_pending"):
            if not self._dependencies_met(info):
                continue

            # Exponential backoff: don't resubmit too quickly after failure
            if info["status"] == "retry_pending":
                retry_after = info.get("retry_after")
                if retry_after and time.time() < retry_after:
                    continue

            try:
                job_id = _sbatch(key, info)
                info["status"] = "submitted"
                info["slurm_job_id"] = job_id
                info["submitted"] = now_iso()
                info["attempts"] = info.get("attempts", 0) + 1
                submitted += 1
            except RuntimeError as e:
                log.error("Failed to submit %s: %s", key, e)
                info["status"] = "failed"
                info["failure_reason"] = f"SUBMIT_ERROR: {e}"

        return submitted

    # ----- Failure handling -----

    def _handle_failures(self) -> int:
        """React to failed stages with backoff. Returns count of retries scheduled."""
        retried = 0
        for key, info in list(self._stages_with_status("failed")):
            reason = info.get("failure_reason", "FAILED")
            reaction = FAILURE_REACTIONS.get(reason, FAILURE_REACTIONS["FAILED"])
            max_retries = reaction.get("max_retries", info.get("max_retries", 1))

            if info.get("attempts", 1) > max_retries:
                update_stage(
                    self.state,
                    key,
                    "abandoned",
                    self.state_path,
                    reason=f"{reason} after {info['attempts']} attempts",
                )
                log.error(
                    "Stage %s ABANDONED after %d attempts (%s)", key, info["attempts"], reason
                )
                continue

            if reaction["action"] == "retry":
                resources = info.get("resources", {})
                if reaction.get("adjust_mem"):
                    resources["mem"] = _scale_mem(
                        resources.get("mem", "16G"), reaction["adjust_mem"]
                    )
                if reaction.get("adjust_time"):
                    resources["time"] = _scale_time(
                        resources.get("time", "3:00:00"), reaction["adjust_time"]
                    )
                if reaction.get("exclude_node") and info.get("failed_node"):
                    info.setdefault("exclude_nodes", []).append(info["failed_node"])
                info["resources"] = resources

                # Phase 2: Checkpoint-aware TIMEOUT resume
                cli = info.get("cli_args", {})
                if reason == "TIMEOUT":
                    from graphids.config.paths import EXPERIMENT_ROOT, run_id_str

                    rid = run_id_str(
                        cli.get("dataset", ""),
                        cli.get("model", "vgae"),
                        cli.get("scale", self.scale),
                        cli.get("stage", ""),
                    )
                    ckpt = _find_checkpoint(EXPERIMENT_ROOT, rid)
                    if ckpt:
                        cli["ckpt_path"] = ckpt
                        log.info("Stage %s: found checkpoint for resume: %s", key, ckpt)
                    else:
                        log.info("Stage %s: no checkpoint found, restarting from scratch", key)

                # Exponential backoff before next submission
                attempt = info.get("attempts", 1)
                delay = _retry_delay(attempt)
                info["retry_after"] = time.time() + delay
                info["status"] = "retry_pending"
                retried += 1
                log.warning(
                    "Stage %s: %s → retry #%d in %.0fs (mem=%s, time=%s)",
                    key,
                    reason,
                    attempt + 1,
                    delay,
                    resources.get("mem"),
                    resources.get("time"),
                )

            elif reaction["action"] == "pause":
                update_stage(
                    self.state,
                    key,
                    "paused",
                    self.state_path,
                    reason=f"User cancelled (job {info.get('slurm_job_id')})",
                )
                log.info("Stage %s PAUSED (user cancelled)", key)

        return retried

    # ----- Deadlock detection -----

    def _detect_deadlock(self) -> bool:
        """Check for deadlock and abandon unrecoverable stages. Returns True to break loop."""
        active = list(self._stages_with_status("submitted", "running"))
        pending_ready = [
            k
            for k, s in self._stages_with_status("pending", "retry_pending")
            if self._dependencies_met(s)
        ]
        if active or pending_ready or self._all_terminal():
            return False

        blocked = [
            k
            for k, s in self._stages_with_status("pending", "retry_pending")
            if not self._dependencies_met(s)
        ]
        if not blocked:
            return False

        # Check if any blocked stages have permanently-failed dependencies
        unrecoverable = [
            (bk, dep)
            for bk in blocked
            for dep in self.stages[bk].get("depends_on", [])
            if self.stages.get(dep, {}).get("status") in ("abandoned", "paused")
        ]

        if unrecoverable:
            for bk, dep in unrecoverable:
                log.error(
                    "Stage %s blocked by %s (%s) — abandoning", bk, dep, self.stages[dep]["status"]
                )
                update_stage(
                    self.state,
                    bk,
                    "abandoned",
                    self.state_path,
                    reason=f"dependency {dep} is {self.stages[dep]['status']}",
                )
            return False  # keep looping, we just freed some stages

        log.warning("Deadlock: %d stages blocked on unmet dependencies", len(blocked))
        return True

    # ----- Artifact verification -----

    def _verify_completions(self) -> int:
        """Verify artifacts exist for completed (unverified) stages."""
        from graphids.config import get_resolver
        from graphids.config.resolver import resolve

        resolver = get_resolver()
        failed = 0

        for key, info in list(self._stages_with_status("completed")):
            if info.get("verified"):
                continue

            cli = info.get("cli_args", {})
            stage = cli.get("stage", "")

            # Evaluation doesn't produce best_model.pt
            if stage == "evaluation":
                info["verified"] = True
                continue

            try:
                cfg = resolve(
                    cli.get("model", "vgae"),
                    cli.get("scale", self.scale),
                    dataset=cli.get("dataset", ""),
                    seed=cli.get("seed", 42),
                )
            except Exception as e:
                log.warning("Cannot verify %s — config resolve failed: %s", key, e)
                info["verified"] = True
                continue

            required = ["best_model.pt", "config.json", "metrics.json"]
            missing = [
                n
                for n in required
                if not resolver.exists(cfg, stage, n, model_type=cli.get("model"))
            ]

            if missing:
                log.warning(
                    "Stage %s COMPLETED but artifacts missing: %s — marking failed", key, missing
                )
                info["status"] = "failed"
                info["failure_reason"] = "MISSING_ARTIFACTS"
                failed += 1
            else:
                info["verified"] = True
                log.info("Stage %s artifacts verified", key)

        return failed

    # ----- Exit hooks -----

    def _run_exit_hooks(self, success: bool) -> None:
        """Run post-pipeline hooks (HF push, notifications, etc.)."""
        if not self.exit_hooks:
            return

        status_word = "SUCCESS" if success else "PARTIAL_FAILURE"
        log.info("Running %d exit hook(s) (pipeline %s)", len(self.exit_hooks), status_word)

        for hook_cmd in self.exit_hooks:
            try:
                env = {**os.environ, "KD_GAT_PIPELINE_STATUS": status_word}
                result = subprocess.run(
                    hook_cmd, capture_output=True, text=True, timeout=300, env=env
                )
                if result.returncode != 0:
                    log.warning("Exit hook %s failed: %s", hook_cmd[0], result.stderr.strip()[:200])
                else:
                    log.info("Exit hook %s completed", hook_cmd[0])
            except Exception as e:
                log.warning("Exit hook %s error: %s", hook_cmd[0], e)

    # ----- Validation -----

    def _validate_upfront(self) -> None:
        """Fail fast with actionable errors before entering control loop."""
        errors: list[str] = []

        if not os.environ.get("KD_GAT_SLURM_ACCOUNT") and SLURM_ACCOUNT == "PAS1266":
            log.info("Using default SLURM account: %s", SLURM_ACCOUNT)

        for tool in ("sbatch", "sacct"):
            result = subprocess.run(["which", tool], capture_output=True, text=True)
            if result.returncode != 0:
                errors.append(f"{tool} not found — are you on a SLURM cluster?")

        from graphids.pipeline.validate import validate_datasets

        errors.extend(validate_datasets(self.datasets, self.scale))

        cache_dir = self.state_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            free_gb = shutil.disk_usage(cache_dir).free / (1024**3)
            if free_gb < 5:
                errors.append(f"Low disk space: {free_gb:.1f} GB free at {cache_dir}")
        except OSError:
            pass

        (PROJECT_ROOT / "slurm_logs").mkdir(parents=True, exist_ok=True)

        if errors:
            raise ValueError("Pre-flight validation failed:\n  " + "\n  ".join(errors))

        log.info(
            "Pre-flight validation passed (%d datasets, %d seeds, %d stages)",
            len(self.datasets),
            len(self.seeds),
            len(self.stages),
        )

    # ----- Display -----

    def _print_plan(self) -> None:
        """Dry-run: show what would be submitted."""
        log.info("=== Coordinator Dry Run ===")
        log.info("Datasets: %s", self.datasets)
        log.info("Seeds: %s", self.seeds)
        log.info("Scale: %s", self.scale)
        log.info("")

        by_dataset: dict[str, list[tuple[str, dict]]] = {}
        for key, info in sorted(self.stages.items()):
            by_dataset.setdefault(key.split("/")[0], []).append((key, info))

        for dataset, stage_list in by_dataset.items():
            log.info("--- %s ---", dataset)
            for key, info in stage_list:
                res = info.get("resources", {})
                deps = info.get("depends_on", [])
                dep_str = f" (after: {', '.join(d.split('/')[-2] for d in deps)})" if deps else ""
                log.info(
                    "  %-50s  %s/%s  mem=%-4s time=%-8s gpu=%d%s",
                    key,
                    res.get("partition", "?"),
                    res.get("cpus", "?"),
                    res.get("mem", "?"),
                    res.get("time", "?"),
                    res.get("gpu", 0),
                    dep_str,
                )

        total_gpu = sum(
            _time_to_hours(s.get("resources", {}).get("time", "0:00:00"))
            for s in self.stages.values()
            if s.get("resources", {}).get("gpu", 0) > 0
        )
        total_cpu = sum(
            _time_to_hours(s.get("resources", {}).get("time", "0:00:00"))
            for s in self.stages.values()
            if s.get("resources", {}).get("gpu", 0) == 0
        )
        log.info("")
        log.info("Estimated max GPU hours: %.1f", total_gpu)
        log.info("Estimated max CPU hours: %.1f", total_cpu)
        log.info("Total stages: %d", len(self.stages))
        log.info("=== End Dry Run ===")

    def _log_summary(self) -> None:
        counts: dict[str, int] = {}
        for info in self.stages.values():
            status = info.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        parts = [f"{s}={c}" for s, c in sorted(counts.items())]
        log.info("Status: %s (total=%d)", ", ".join(parts), len(self.stages))

    # ----- Phase 4: Resource profiling -----

    def _save_resource_profile(self) -> None:
        """Save actual resource usage to a JSONL file for right-sizing future runs.

        Each line: {stage_type, dataset, actual_resources, requested_resources}.
        Accumulated across pipeline runs for historical averaging.
        """

        profile_path = PROJECT_ROOT / ".cache" / "kd-gat" / "resource_profile.jsonl"
        profile_path.parent.mkdir(parents=True, exist_ok=True)

        entries = []
        for key, info in self.stages.items():
            actual = info.get("actual_resources")
            if not actual:
                continue
            cli = info.get("cli_args", {})
            # Stage type key (model_stage, e.g. "vgae_autoencoder")
            stage_type = f"{cli.get('model', 'unknown')}_{cli.get('stage', 'unknown')}"
            entries.append(
                {
                    "stage_type": stage_type,
                    "scale": cli.get("scale", self.scale),
                    "dataset": cli.get("dataset", ""),
                    "requested": info.get("resources", {}),
                    "actual": actual,
                    "timestamp": info.get("completed", now_iso()),
                }
            )

        if entries:
            with open(profile_path, "a") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
            log.info("Saved %d resource profiles to %s", len(entries), profile_path)
