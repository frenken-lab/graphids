"""Submit-and-poll driver for pipeline orchestration.

Not a daemon — a script you run once. It:
1. Loads/generates a plan → inserts into SQLite
2. Finds jobs whose parents are COMPLETED and that aren't yet submitted
3. Submits them via the executor, records native IDs
4. Polls scheduler every N seconds, updates transitions
5. Handles failures with retry logic
6. Repeats until all jobs are terminal
7. Prints summary, exits

Ctrl-C safe: state is in SQLite, restart picks up where it left off.

Can also run in fire-and-forget mode: submit everything with --dependency
upfront and exit immediately (no polling loop).

Usage (via cli.py):
    python -m graphids.pipeline.cli orchestrate --dataset hcrl_sa --seeds 42,123,456
    python -m graphids.pipeline.cli orchestrate --dataset hcrl_sa --dry-run
    python -m graphids.pipeline.cli orchestrate --resume <run_id>
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphids.config.constants import PROJECT_ROOT

from .executor import DryRunExecutor, JobExecutor
from .job import JobSpec, JobState
from .planner import build_plan, print_plan
from .store import PipelineStore

log = logging.getLogger(__name__)

# Failure handling
_MAX_RETRIES = 2
_RETRY_REASONS = {"FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "PREEMPTED"}
_MEM_SCALE_REASONS = {"OUT_OF_MEMORY"}
_TIME_SCALE_REASONS = {"TIMEOUT"}


class PipelineDriver:
    """Orchestration driver: submit-and-poll loop with SQLite state."""

    def __init__(
        self,
        store: PipelineStore,
        executor: JobExecutor,
        run_id: str,
        poll_interval: int = 30,
        max_retries: int = _MAX_RETRIES,
        fire_and_forget: bool = False,
    ):
        self.store = store
        self.executor = executor
        self.run_id = run_id
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.fire_and_forget = fire_and_forget
        # Track native_id → job_id + attempt_id for polling
        self._active: dict[str, tuple[str, int]] = {}
        self._rebuild_active_tracking()

    def _rebuild_active_tracking(self) -> None:
        """Rebuild active job tracking from DB (for resume)."""
        states = self.store.current_states(self.run_id)
        for job_id, state in states.items():
            if state in (JobState.QUEUED, JobState.RUNNING):
                attempt = self.store.latest_attempt(job_id)
                if attempt and attempt["native_id"]:
                    self._active[attempt["native_id"]] = (job_id, attempt["attempt_id"])

    # ── Main entry points ───────────────────────────────────────────────

    def run(self) -> bool:
        """Main control loop. Returns True if all jobs completed successfully."""
        log.info("Starting driver loop (poll every %ds)", self.poll_interval)
        iteration = 0

        while True:
            iteration += 1
            n_submitted = self._submit_ready()
            n_polled = self._poll_active()
            self._propagate_failures()

            if iteration % 10 == 1 or n_submitted or n_polled:
                self._log_summary()

            if self._all_terminal():
                break

            if self._detect_deadlock():
                log.error("Deadlock detected — no jobs can make progress")
                break

            time.sleep(self.poll_interval)

        self._log_summary()
        summary = self.store.summary(self.run_id)
        failed = summary.get("failed", 0) + summary.get("abandoned", 0)
        if failed:
            log.error("Pipeline finished with %d failed/abandoned jobs", failed)
            return False
        log.info("Pipeline complete — all jobs succeeded.")
        return True

    def submit_all_with_deps(self) -> None:
        """Fire-and-forget mode: submit all jobs upfront with scheduler dependencies.

        Jobs are submitted in topological order. Each job's --dependency flag
        references the native IDs of its parent jobs. After submission, the
        driver exits — the scheduler handles execution ordering.
        """
        from graphlib import TopologicalSorter

        states = self.store.current_states(self.run_id)
        all_jobs = {
            j["id"]: j
            for j in self.store.jobs_in_state(self.run_id, JobState.PENDING, JobState.FAILED)
        }

        # Build topo sort graph
        graph: dict[str, set[str]] = {}
        for jid, job in all_jobs.items():
            parents = json.loads(job["parents"])
            # Only include parents that are also in our submission set
            graph[jid] = {p for p in parents if p in all_jobs}
            # Parents already completed don't need to be in the graph
            for p in parents:
                if p not in all_jobs and states.get(p) == JobState.COMPLETED:
                    pass  # already done, no edge needed

        native_ids: dict[str, str] = {}  # job_id → native_id

        # Also load native IDs for already-completed jobs
        for jid, state in states.items():
            if state == JobState.COMPLETED:
                attempt = self.store.latest_attempt(jid)
                if attempt and attempt["native_id"]:
                    native_ids[jid] = attempt["native_id"]

        ts = TopologicalSorter(graph)
        ts.prepare()

        submitted = 0
        while ts.is_active():
            for jid in ts.get_ready():
                job_row = all_jobs.get(jid)
                if not job_row:
                    ts.done(jid)
                    continue

                job_spec = self._row_to_jobspec(job_row)
                parent_ids = json.loads(job_row["parents"])
                dep_native = [native_ids[p] for p in parent_ids if p in native_ids]

                native_id = self.executor.submit(job_spec, dependency_ids=dep_native or None)
                native_ids[jid] = native_id

                attempt_id = self.store.create_attempt(jid, native_id)
                self.store.transition(jid, JobState.QUEUED, attempt_id)
                submitted += 1
                ts.done(jid)

        log.info("Fire-and-forget: submitted %d jobs with scheduler dependencies", submitted)

    # ── Internal: submit/poll/retry ─────────────────────────────────────

    def _submit_ready(self) -> int:
        """Find and submit jobs whose parents are all COMPLETED."""
        ready = self.store.ready_jobs(self.run_id)
        submitted = 0

        for job_row in ready:
            job_id = job_row["id"]

            # Check retry limit
            attempts = self.store.attempt_count(job_id)
            if attempts > self.max_retries:
                self.store.transition(job_id, JobState.ABANDONED, detail="max retries exceeded")
                continue

            job_spec = self._row_to_jobspec(job_row)

            # For retries after OOM/TIMEOUT, scale resources
            if attempts > 0:
                job_spec = self._scale_for_retry(job_spec, job_id)

            try:
                native_id = self.executor.submit(job_spec)
            except Exception as e:
                log.error("Failed to submit %s: %s", job_row["name"], e)
                self.store.transition(job_id, JobState.FAILED, detail=str(e))
                continue

            attempt_id = self.store.create_attempt(job_id, native_id)
            self.store.transition(job_id, JobState.QUEUED, attempt_id)
            self._active[native_id] = (job_id, attempt_id)
            submitted += 1

        return submitted

    def _poll_active(self) -> int:
        """Poll all active jobs for status changes."""
        changes = 0
        done_natives = []

        for native_id, (job_id, attempt_id) in self._active.items():
            state, meta = self.executor.poll(native_id)
            current = self.store.current_state(job_id)

            if state == current:
                continue

            # Update attempt metadata
            updates: dict[str, Any] = {}
            if meta.get("hostname"):
                updates["hostname"] = meta["hostname"]
            if state == JobState.RUNNING:
                updates["started_at"] = datetime.now(UTC).isoformat()
            if state.is_terminal:
                updates["finished_at"] = datetime.now(UTC).isoformat()
                updates["failure_reason"] = meta.get("failure_reason")
                if meta.get("max_rss") or meta.get("elapsed"):
                    updates["resources_used"] = json.dumps(
                        {k: v for k, v in meta.items() if k in ("max_rss", "elapsed", "exit_code")}
                    )
            if updates:
                self.store.update_attempt(attempt_id, **updates)

            # Record state transition
            if state == JobState.FAILED:
                reason = meta.get("failure_reason", "FAILED")
                # Check if retryable
                if (
                    reason in _RETRY_REASONS
                    and self.store.attempt_count(job_id) <= self.max_retries
                ):
                    # Reset to PENDING for retry
                    self.store.transition(job_id, JobState.FAILED, attempt_id, detail=reason)
                    self.store.transition(job_id, JobState.PENDING, detail=f"retry after {reason}")
                    done_natives.append(native_id)
                else:
                    self.store.transition(job_id, JobState.FAILED, attempt_id, detail=reason)
                    done_natives.append(native_id)
            else:
                self.store.transition(job_id, state, attempt_id)
                if state.is_terminal:
                    done_natives.append(native_id)

            changes += 1

        for nid in done_natives:
            del self._active[nid]

        return changes

    def _propagate_failures(self) -> None:
        """Mark PENDING jobs as ABANDONED if any parent is in a terminal failure state."""
        pending = self.store.jobs_in_state(self.run_id, JobState.PENDING)
        for job_row in pending:
            if self.store.has_failed_parents(job_row["id"], self.run_id):
                self.store.transition(
                    job_row["id"],
                    JobState.ABANDONED,
                    detail="parent job failed",
                )

    def _scale_for_retry(self, job_spec: JobSpec, job_id: str) -> JobSpec:
        """Scale resources based on previous failure reason."""
        attempt = self.store.latest_attempt(job_id)
        if not attempt or not attempt["failure_reason"]:
            return job_spec

        reason = attempt["failure_reason"]
        resources = job_spec.resources
        if reason in _MEM_SCALE_REASONS:
            resources = resources.scale_memory(2.0)
            log.info("Scaling memory for retry: %s → %dGB", job_spec.name, resources.memory_gb)
        if reason in _TIME_SCALE_REASONS:
            resources = resources.scale_walltime(1.5)
            log.info("Scaling walltime for retry: %s → %s", job_spec.name, resources.walltime_str)
        return job_spec.with_resources(resources)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _all_terminal(self) -> bool:
        states = self.store.current_states(self.run_id)
        return all(s.is_terminal for s in states.values()) and len(states) > 0

    def _detect_deadlock(self) -> bool:
        """Detect if no jobs can make progress."""
        states = self.store.current_states(self.run_id)
        if not states:
            return False
        non_terminal = {jid for jid, s in states.items() if not s.is_terminal}
        if not non_terminal:
            return False  # all terminal = done, not deadlock
        # If there are non-terminal jobs but none are QUEUED/RUNNING and none are ready
        active = any(s in (JobState.QUEUED, JobState.RUNNING) for s in states.values())
        ready = self.store.ready_jobs(self.run_id)
        return not active and not ready and bool(non_terminal)

    def _row_to_jobspec(self, row: dict) -> JobSpec:
        """Convert a DB row back to a JobSpec."""
        from uuid import UUID as _UUID

        resources_data = json.loads(row["resources"])
        # Handle timedelta serialization
        if "walltime" in resources_data and isinstance(resources_data["walltime"], (int, float)):
            from datetime import timedelta

            resources_data["walltime"] = timedelta(seconds=resources_data["walltime"])

        return JobSpec(
            id=_UUID(row["id"]),
            name=row["name"],
            executable=row["executable"],
            arguments=json.loads(row["arguments"]),
            parameters=json.loads(row["parameters"]),
            resources=_parse_resource_spec(resources_data),
            parents=[_UUID(p) for p in json.loads(row["parents"])],
            environment=json.loads(row["environment"]),
            tags=json.loads(row["tags"]),
        )

    def _log_summary(self) -> None:
        summary = self.store.summary(self.run_id)
        total = self.store.total_jobs(self.run_id)
        parts = " | ".join(f"{k}={v}" for k, v in sorted(summary.items()))
        log.info("Pipeline status [%d jobs]: %s", total, parts)


def _parse_resource_spec(data: dict) -> Any:
    """Parse ResourceSpec from JSON, handling timedelta."""
    from datetime import timedelta

    from .job import ResourceSpec

    if "walltime" in data:
        wt = data["walltime"]
        if isinstance(wt, (int, float)):
            data = {**data, "walltime": timedelta(seconds=wt)}
        elif isinstance(wt, str) and ":" in wt:
            parts = wt.split(":")
            if len(parts) == 3:
                data = {
                    **data,
                    "walltime": timedelta(
                        hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2])
                    ),
                }
    return ResourceSpec(**data)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_orchestrate(
    datasets: list[str],
    seeds: list[int],
    variant_filter: list[str] | None = None,
    poll_interval: int = 30,
    dry_run: bool = False,
    fire_and_forget: bool = False,
    db_path: str | Path | None = None,
    resume_run: str | None = None,
    backend: str | None = None,
) -> bool:
    """Top-level entry point for pipeline orchestration.

    Returns True if pipeline completed successfully.
    """
    from graphids.config.resolver import resolve

    db_uri = db_path or os.getenv(
        "KD_GAT_DB_URI",
        f"sqlite:///{PROJECT_ROOT / '.cache' / 'kd-gat' / 'pipeline.db'}",
    )

    with PipelineStore(db_uri) as store:
        if resume_run:
            run_id = resume_run
            total = store.total_jobs(run_id)
            if total == 0:
                log.error("No jobs found for run_id=%s", run_id)
                return False
            log.info("Resuming run %s (%d jobs)", run_id, total)
        else:
            # Resolve variants from config
            cfg = resolve("vgae", "large")
            all_variants = [
                {
                    "name": v.name,
                    "scale": v.scale,
                    "auxiliaries": v.auxiliaries,
                    "needs_teacher": v.needs_teacher,
                    "stages": list(v.stages),
                }
                for v in cfg.variants
            ]
            if variant_filter:
                known = {v["name"] for v in all_variants}
                unknown = set(variant_filter) - known
                if unknown:
                    raise ValueError(f"Unknown variant(s): {unknown}. Available: {known}")
                all_variants = [v for v in all_variants if v["name"] in variant_filter]

            # Build plan
            jobs = build_plan(datasets, seeds, all_variants)

            if dry_run:
                print_plan(jobs)
                # Also test with DryRunExecutor
                run_id = f"dry_run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
                store.create_run(run_id, {"datasets": datasets, "seeds": seeds, "dry_run": True})
                store.insert_jobs(run_id, jobs)
                executor = DryRunExecutor()
                driver = PipelineDriver(store, executor, run_id, poll_interval=0)
                driver.submit_all_with_deps()
                summary = store.summary(run_id)
                log.info("Dry run complete: %s", summary)
                return True

            # Create run and insert jobs
            run_id = f"{'_'.join(datasets)}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
            store.create_run(
                run_id,
                {
                    "datasets": datasets,
                    "seeds": seeds,
                    "variants": [v["name"] for v in all_variants],
                },
            )
            n_inserted = store.insert_jobs(run_id, jobs)
            log.info("Created run %s: %d jobs (%d new)", run_id, len(jobs), n_inserted)

        # Create executor
        executor = JobExecutor.create(backend)

        # Run
        driver = PipelineDriver(
            store,
            executor,
            run_id,
            poll_interval=poll_interval,
            fire_and_forget=fire_and_forget,
        )

        if fire_and_forget:
            driver.submit_all_with_deps()
            return True
        else:
            return driver.run()
