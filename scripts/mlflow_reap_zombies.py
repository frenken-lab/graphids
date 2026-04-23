#!/usr/bin/env python
"""Reap zombie MLflow runs — RUNNING rows left by SLURM jobs that died
without triggering ``on_fit_end``. Cross-references each run's
``slurm.slurm_job_id`` tag against sacct and flips terminated jobs to
FINISHED/FAILED/KILLED.

Usage:
    python scripts/mlflow_reap_zombies.py            # preview
    python scripts/mlflow_reap_zombies.py --apply    # write changes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime

import mlflow
from mlflow.tracking import MlflowClient

from graphids._mlflow import ensure_tracking_uri

# sacct State → MLflow terminal status. Missing keys fall through to KILLED.
_SLURM_TO_MLFLOW = {
    "COMPLETED": "FINISHED",
    "FAILED": "FAILED",
    "OUT_OF_MEMORY": "FAILED",
    "NODE_FAIL": "FAILED",
    "CANCELLED": "KILLED",
    "TIMEOUT": "KILLED",
    "PREEMPTED": "KILLED",
}
# sacct states that mean "leave alone" — the run really is still running.
_ALIVE = {"RUNNING", "PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED"}


def _sacct_states(jids: list[str]) -> dict[str, tuple[str, int | None]]:
    """``{jid: (state, end_time_ms_or_None)}`` via one batched sacct call."""
    out = subprocess.check_output(
        ["sacct", "-j", ",".join(jids), "-X", "-n", "-P", "--format=JobID,State,End"],
        text=True,
    )
    result: dict[str, tuple[str, int | None]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jid, state, end = (p.strip() for p in parts[:3])
        # sacct emits e.g. "CANCELLED by 12345" — drop the qualifier.
        state = state.split()[0] if state else ""
        end_ms: int | None = None
        if end and end != "Unknown":
            try:
                end_ms = int(
                    datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000
                )
            except ValueError:
                pass
        result[jid] = (state, end_ms)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="actually write changes (default: dry-run)")
    args = p.parse_args()

    uri = ensure_tracking_uri()
    if uri is None:
        print(
            "error: ensure_tracking_uri() returned None — is GRAPHIDS_LAKE_ROOT set?",
            file=sys.stderr,
        )
        return 2
    mlflow.set_tracking_uri(uri)
    c = MlflowClient()

    exps = [e.experiment_id for e in c.search_experiments()]
    running = c.search_runs(
        experiment_ids=exps,
        filter_string="attributes.status = 'RUNNING'",
        max_results=10_000,
    )
    print(f"Tracking URI: {uri}")
    print(f"Found {len(running)} RUNNING rows across {len(exps)} experiments")

    candidates: list[tuple[str, str]] = []
    for r in running:
        jid = r.data.tags.get("slurm.slurm_job_id")
        if jid:
            candidates.append((r.info.run_id, jid))
    no_tag = len(running) - len(candidates)
    print(f"  tagged with slurm.slurm_job_id: {len(candidates)} (skipped {no_tag} without tag)")
    if not candidates:
        return 0

    states = _sacct_states([j for _, j in candidates])
    actions: list[tuple[str, str, str, int | None]] = []
    for run_id, jid in candidates:
        info = states.get(jid)
        if info is None:
            continue  # sacct has no record — likely purged history, leave alone
        state, end_ms = info
        if state in _ALIVE:
            continue
        actions.append((run_id, jid, _SLURM_TO_MLFLOW.get(state, "KILLED"), end_ms))

    print(f"\nWould reap {len(actions)} zombie runs:")
    for run_id, jid, status, end_ms in actions[:20]:
        end_str = datetime.fromtimestamp(end_ms / 1000).isoformat() if end_ms else "?"
        print(f"  {run_id[:8]} jid={jid:<10} -> {status:<8} (end={end_str})")
    if len(actions) > 20:
        print(f"  ... and {len(actions) - 20} more")

    if not args.apply:
        print("\nDry-run. Pass --apply to write changes.")
        return 0

    for run_id, _, status, end_ms in actions:
        c.set_terminated(run_id, status=status, end_time=end_ms)
    print(f"\nReaped {len(actions)} runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
