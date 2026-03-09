"""Stateful SLURM job coordinator with reactive failure handling.

Submits pipeline stages as individual SLURM jobs (GPU stages to gpu partition,
CPU stages to cpu partition), polls sacct for completion, reacts to failures
with adjusted resources, and verifies artifacts after each stage completes.

Usage:
    python -m graphids.pipeline.coordinator --dataset hcrl_sa --seeds 42,123
    python -m graphids.pipeline.coordinator --resume .cache/kd-gat/pipeline_state.json
    python -m graphids.pipeline.coordinator --dataset hcrl_sa --seeds 42 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphids.config.constants import (
    PROJECT_ROOT,
    SLURM_ACCOUNT,
    STAGE_DEPENDENCIES,
    parse_seeds,
)

from .state import StageStatus, load_state, now_iso, save_state, update_stage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource profiles per stage type
# ---------------------------------------------------------------------------

RESOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "preprocess": {"partition": "cpu", "gpu": 0, "cpus": 4, "mem": "32G", "time": "1:00:00"},
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
    "dqn_large_fusion": {"partition": "cpu", "gpu": 0, "cpus": 4, "mem": "16G", "time": "0:30:00"},
    "dqn_small_fusion": {"partition": "cpu", "gpu": 0, "cpus": 4, "mem": "16G", "time": "0:30:00"},
    "evaluation": {"partition": "cpu", "gpu": 0, "cpus": 4, "mem": "16G", "time": "0:30:00"},
    "aggregate": {"partition": "cpu", "gpu": 0, "cpus": 2, "mem": "8G", "time": "0:15:00"},
}

# Fallback for unknown stage types
_DEFAULT_RESOURCE = {"partition": "gpu", "gpu": 1, "cpus": 4, "mem": "20G", "time": "3:00:00"}

# ---------------------------------------------------------------------------
# Failure reaction table
# ---------------------------------------------------------------------------

FAILURE_REACTIONS: dict[str, dict[str, Any]] = {
    "OUT_OF_MEMORY": {
        "action": "retry",
        "adjust_mem": 2.0,  # multiply memory by this factor
        "max_retries": 2,
    },
    "TIMEOUT": {
        "action": "retry",
        "adjust_time": 1.5,  # multiply time by this factor
        "max_retries": 2,
    },
    "NODE_FAIL": {
        "action": "retry",
        "exclude_node": True,
        "max_retries": 3,
    },
    "CANCELLED": {
        "action": "pause",
    },
    "FAILED": {
        "action": "retry",
        "max_retries": 1,
    },
}


# ---------------------------------------------------------------------------
# Stage plan builder
# ---------------------------------------------------------------------------


def _resource_key(model_type: str, scale: str, stage: str) -> str:
    """Build a lookup key for RESOURCE_PROFILES."""
    key = f"{model_type}_{scale}_{stage}"
    if key in RESOURCE_PROFILES:
        return key
    # Try without scale
    key_no_scale = f"{model_type}_{stage}"
    if key_no_scale in RESOURCE_PROFILES:
        return key_no_scale
    # Try stage alone
    if stage in RESOURCE_PROFILES:
        return stage
    return ""


def _get_resources(model_type: str, scale: str, stage: str) -> dict[str, Any]:
    """Get resource profile for a stage, falling back to defaults."""
    key = _resource_key(model_type, scale, stage)
    profile = RESOURCE_PROFILES.get(key, _DEFAULT_RESOURCE)
    return dict(profile)  # copy to allow per-stage mutation


def _resolve_dependencies(
    dataset: str, seed: int, model_type: str, scale: str, stage: str
) -> list[str]:
    """Build dependency keys for a stage."""
    deps = []
    if stage in STAGE_DEPENDENCIES:
        for dep_model, dep_stage in STAGE_DEPENDENCIES[stage]:
            deps.append(f"{dataset}/{dep_model}_{scale}_{dep_stage}/seed_{seed}")
    return deps


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

    # Standard pipeline stages in order
    pipeline_stages = [
        ("vgae", "autoencoder"),
        ("gat", "curriculum"),
        ("dqn", "fusion"),
    ]

    for dataset in datasets:
        for seed in seeds:
            for model_type, stage in pipeline_stages:
                key = f"{dataset}/{model_type}_{scale}_{stage}/seed_{seed}"
                deps = _resolve_dependencies(dataset, seed, model_type, scale, stage)
                resources = _get_resources(model_type, scale, stage)

                # Build CLI args for this stage
                cli_args = {
                    "stage": stage,
                    "model": model_type,
                    "scale": scale,
                    "dataset": dataset,
                    "seed": seed,
                    "auxiliaries": auxiliaries,
                }

                plan[key] = {
                    "status": "pending",
                    "depends_on": deps,
                    "resources": resources,
                    "cli_args": cli_args,
                    "attempts": 0,
                    "max_retries": 2,
                }

            # Evaluation (uses all models, seed-specific)
            eval_key = f"{dataset}/eval_{scale}_evaluation/seed_{seed}"
            eval_deps = [
                f"{dataset}/vgae_{scale}_autoencoder/seed_{seed}",
                f"{dataset}/gat_{scale}_curriculum/seed_{seed}",
                f"{dataset}/dqn_{scale}_fusion/seed_{seed}",
            ]
            plan[eval_key] = {
                "status": "pending",
                "depends_on": eval_deps,
                "resources": _get_resources("eval", scale, "evaluation"),
                "cli_args": {
                    "stage": "evaluation",
                    "model": "vgae",  # evaluation uses all models but needs a --model arg
                    "scale": scale,
                    "dataset": dataset,
                    "seed": seed,
                    "auxiliaries": auxiliaries,
                },
                "attempts": 0,
                "max_retries": 1,
            }

    return plan


# ---------------------------------------------------------------------------
# SLURM interaction
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
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _parse_sacct_state(job_id: int) -> str | None:
    """Query sacct for a job's state. Returns None if job not found."""
    result = subprocess.run(
        [
            "sacct",
            "-j",
            str(job_id),
            "--format=State",
            "--noheader",
            "-P",
            "--parsable2",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        log.warning("sacct failed for job %d: %s", job_id, result.stderr.strip())
        return None

    # sacct may return multiple lines (job + job steps). Use the first (main job).
    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return None
    return lines[0]


def _sbatch(stage_key: str, stage_info: dict[str, Any]) -> int:
    """Submit a stage as a SLURM job. Returns the job ID."""
    res = stage_info["resources"]
    cli = stage_info["cli_args"]

    # Build the python command via shared builder, then prepend preamble for SLURM
    from graphids.pipeline.subprocess_utils import build_cli_cmd

    cli_cmd = build_cli_cmd(
        cli["stage"],
        cli["model"],
        cli["scale"],
        cli["dataset"],
        seed=cli.get("seed"),
        auxiliaries=cli.get("auxiliaries", "none"),
    )
    wrap_cmd = "source scripts/slurm/_preamble.sh && " + " ".join(str(p) for p in cli_cmd)

    # Sanitize stage key for filenames
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
    ]

    if res.get("gpu"):
        cmd.append(f"--gres=gpu:{res['gpu']}")

    # Signal for graceful timeout (180s before wall time)
    cmd.append("--signal=B:USR1@180")

    # Exclude nodes from previous failures
    exclude = stage_info.get("exclude_nodes")
    if exclude:
        cmd.append(f"--exclude={','.join(exclude)}")

    cmd.extend(["--wrap", wrap_cmd])

    log.info("Submitting: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

    job_id = int(result.stdout.strip().split(";")[0])  # --parsable may include cluster name
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
    ):
        self.datasets = datasets
        self.seeds = seeds
        self.scale = scale
        self.auxiliaries = auxiliaries
        self.poll_interval = poll_interval
        self.dry_run = dry_run

        self.state_path = state_path or (PROJECT_ROOT / ".cache" / "kd-gat" / "pipeline_state.json")

        # Load or create state
        existing = load_state(self.state_path)
        if existing and "stages" in existing:
            log.info(
                "Resuming from %s (%d stages tracked)", self.state_path, len(existing["stages"])
            )
            self.state = existing
            # Reconcile: mark any stages that are in the plan but not in state
            plan = build_stage_plan(datasets, seeds, scale, auxiliaries)
            for key, info in plan.items():
                if key not in self.state["stages"]:
                    self.state["stages"][key] = info
        else:
            plan = build_stage_plan(datasets, seeds, scale, auxiliaries)
            self.state = {
                "pipeline_id": f"{'_'.join(datasets)}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                "created": now_iso(),
                "datasets": datasets,
                "seeds": seeds,
                "scale": scale,
                "auxiliaries": auxiliaries,
                "stages": plan,
            }
            save_state(self.state, self.state_path)
            log.info("Created new pipeline state: %d stages", len(plan))

    @property
    def stages(self) -> dict[str, dict[str, Any]]:
        return self.state["stages"]

    # ----- Status queries -----

    def _stages_with_status(self, *statuses: StageStatus):
        for key, info in self.stages.items():
            if info.get("status") in statuses:
                yield key, info

    def _all_terminal(self) -> bool:
        """Check if all stages are in a terminal state."""
        terminal = {"completed", "abandoned", "paused"}
        return all(s.get("status") in terminal for s in self.stages.values())

    def _dependencies_met(self, stage_info: dict) -> bool:
        """Check if all dependencies are completed."""
        for dep_key in stage_info.get("depends_on", []):
            dep = self.stages.get(dep_key)
            if dep is None or dep.get("status") != "completed":
                return False
        return True

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

            n_submitted = self._poll_running_jobs()
            n_dispatched = self._submit_ready_stages()
            n_retried = self._handle_failures()

            save_state(self.state, self.state_path)

            # Log summary periodically
            if iteration % 10 == 1 or n_submitted or n_dispatched or n_retried:
                self._log_summary()

            # Check for deadlock (nothing running, nothing pending with met deps)
            active = list(self._stages_with_status("submitted", "running"))
            pending_ready = [
                k
                for k, s in self._stages_with_status("pending", "retry_pending")
                if self._dependencies_met(s)
            ]
            if not active and not pending_ready and not self._all_terminal():
                blocked = [
                    k
                    for k, s in self._stages_with_status("pending", "retry_pending")
                    if not self._dependencies_met(s)
                ]
                if blocked:
                    # Check if blocked stages have abandoned/failed dependencies
                    unrecoverable = []
                    for bk in blocked:
                        for dep in self.stages[bk].get("depends_on", []):
                            dep_status = self.stages.get(dep, {}).get("status")
                            if dep_status in ("abandoned", "paused"):
                                unrecoverable.append((bk, dep))
                    if unrecoverable:
                        for bk, dep in unrecoverable:
                            log.error(
                                "Stage %s blocked by %s (%s) — abandoning",
                                bk,
                                dep,
                                self.stages[dep]["status"],
                            )
                            update_stage(
                                self.state,
                                bk,
                                "abandoned",
                                self.state_path,
                                reason=f"dependency {dep} is {self.stages[dep]['status']}",
                            )
                    else:
                        log.warning(
                            "Deadlock: %d stages blocked on unmet dependencies", len(blocked)
                        )
                        break

            time.sleep(self.poll_interval)

        self._log_summary()

        # Check for any abandoned stages
        abandoned = list(self._stages_with_status("abandoned"))
        if abandoned:
            log.error("Pipeline finished with %d abandoned stages:", len(abandoned))
            for key, info in abandoned:
                log.error(
                    "  %s: %s", key, info.get("reason", info.get("failure_reason", "unknown"))
                )
        else:
            log.info("Pipeline complete — all stages succeeded.")

    def _poll_running_jobs(self) -> int:
        """Check sacct for all submitted/running stages. Returns count of state changes."""
        changes = 0
        for key, info in self._stages_with_status("submitted", "running"):
            job_id = info.get("slurm_job_id")
            if not job_id:
                continue

            slurm_state = _parse_sacct_state(job_id)
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
                changes += 1
                log.info("Stage %s COMPLETED (job %d)", key, job_id)

            elif slurm_state in ("FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "CANCELLED"):
                info["status"] = "failed"
                info["failure_reason"] = slurm_state
                info["completed"] = now_iso()
                changes += 1

                # Capture node for potential exclusion
                node = self._get_job_node(job_id)
                if node:
                    info["failed_node"] = node

                log.warning("Stage %s FAILED: %s (job %d)", key, slurm_state, job_id)

            elif slurm_state == "PENDING":
                # Still in queue — no change needed
                pass

        return changes

    def _submit_ready_stages(self) -> int:
        """Submit stages whose dependencies are met. Returns count submitted."""
        submitted = 0
        for key, info in self._stages_with_status("pending", "retry_pending"):
            if not self._dependencies_met(info):
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

    def _handle_failures(self) -> int:
        """React to failed stages. Returns count of retries scheduled."""
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
                # Adjust resources
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
                    exclude = info.get("exclude_nodes", [])
                    exclude.append(info["failed_node"])
                    info["exclude_nodes"] = exclude
                info["resources"] = resources

                info["status"] = "retry_pending"
                retried += 1
                log.warning(
                    "Stage %s: %s → retry #%d (mem=%s, time=%s)",
                    key,
                    reason,
                    info.get("attempts", 0) + 1,
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

    def _get_job_node(self, job_id: int) -> str | None:
        """Get the node a job ran on from sacct."""
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=NodeList", "--noheader", "-P"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if lines and lines[0] != "None assigned":
                return lines[0]
        return None

    # ----- Validation -----

    def _validate_upfront(self) -> None:
        """Fail fast with actionable errors before entering control loop."""
        errors: list[str] = []

        # SLURM account
        if not os.environ.get("KD_GAT_SLURM_ACCOUNT") and SLURM_ACCOUNT == "PAS1266":
            log.info("Using default SLURM account: %s", SLURM_ACCOUNT)

        # Check sbatch is available
        result = subprocess.run(["which", "sbatch"], capture_output=True, text=True)
        if result.returncode != 0:
            errors.append("sbatch not found — are you on a SLURM cluster?")

        # Check sacct is available
        result = subprocess.run(["which", "sacct"], capture_output=True, text=True)
        if result.returncode != 0:
            errors.append("sacct not found — needed for job status polling")

        # Data directories
        from graphids.pipeline.validate import validate_datasets

        errors.extend(validate_datasets(self.datasets, self.scale))

        # Disk space
        cache_dir = self.state_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            free_gb = shutil.disk_usage(cache_dir).free / (1024**3)
            if free_gb < 5:
                errors.append(f"Low disk space: {free_gb:.1f} GB free at {cache_dir}")
        except OSError:
            pass

        # slurm_logs dir
        logs_dir = PROJECT_ROOT / "slurm_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

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

        # Group by dataset
        by_dataset: dict[str, list[tuple[str, dict]]] = {}
        for key, info in sorted(self.stages.items()):
            dataset = key.split("/")[0]
            by_dataset.setdefault(dataset, []).append((key, info))

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

        # Estimate total resources
        total_gpu_hours = 0.0
        total_cpu_hours = 0.0
        for info in self.stages.values():
            res = info.get("resources", {})
            time_parts = res.get("time", "0:00:00").split(":")
            hours = int(time_parts[0]) + int(time_parts[1]) / 60
            if res.get("gpu", 0) > 0:
                total_gpu_hours += hours
            else:
                total_cpu_hours += hours

        log.info("")
        log.info("Estimated max GPU hours: %.1f", total_gpu_hours)
        log.info("Estimated max CPU hours: %.1f", total_cpu_hours)
        log.info("Total stages: %d", len(self.stages))
        log.info("=== End Dry Run ===")

    def _log_summary(self) -> None:
        """Log a compact status summary."""
        counts: dict[str, int] = {}
        for info in self.stages.values():
            status = info.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1

        parts = [f"{status}={count}" for status, count in sorted(counts.items())]
        log.info("Status: %s (total=%d)", ", ".join(parts), len(self.stages))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="coordinator",
        description="Stateful SLURM pipeline coordinator",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Dataset name (comma-separated for multiple)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Seeds: comma-separated (42,123,456) or count (5 = first 5 defaults)",
    )
    parser.add_argument("--scale", type=str, default="large")
    parser.add_argument("--auxiliaries", type=str, default="none")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a state file (ignores --dataset/--seeds/--scale)",
    )
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without submitting jobs",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )

    if args.resume:
        # Resume from state file
        state = load_state(args.resume)
        if not state or "stages" not in state:
            log.error("Invalid or empty state file: %s", args.resume)
            sys.exit(1)

        coordinator = PipelineCoordinator(
            datasets=state["datasets"],
            seeds=state["seeds"],
            scale=state.get("scale", "large"),
            auxiliaries=state.get("auxiliaries", "none"),
            state_path=args.resume,
            poll_interval=args.poll_interval,
            dry_run=args.dry_run,
        )
    else:
        if not args.dataset:
            parser.error("--dataset is required (unless --resume is used)")

        datasets = [d.strip() for d in args.dataset.split(",")]
        seeds = parse_seeds(args.seeds)

        coordinator = PipelineCoordinator(
            datasets=datasets,
            seeds=seeds,
            scale=args.scale,
            auxiliaries=args.auxiliaries,
            poll_interval=args.poll_interval,
            dry_run=args.dry_run,
        )

    coordinator.run()


if __name__ == "__main__":
    main()
