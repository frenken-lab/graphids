"""SLURM submission + polling helpers."""

from __future__ import annotations

import re
import subprocess
import time as _time
from collections.abc import Callable
from pathlib import Path

from graphids.config.constants import PROJECT_ROOT
from graphids.log import get_logger
from graphids.slurm.core.accounting import sacct_query
from graphids.slurm.env import SLURM_ACCOUNT, SLURM_LOG_DIR
from graphids.slurm.resources import ResourceSpec

log = get_logger(__name__)

StateObserver = Callable[[str, int], None]

_TERMINAL = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "OUT_OF_MEMORY",
        "TIMEOUT",
        "NODE_FAIL",
        "CANCELLED",
        "PREEMPTED",
    }
)


def submit(script: str, resources: ResourceSpec, *, job_name: str, dry_run: bool = False) -> int:
    """Submit sbatch job. Returns job ID (0 if dry_run)."""
    Path(SLURM_LOG_DIR).mkdir(parents=True, exist_ok=True)
    args = [
        "sbatch",
        f"--partition={resources.partition}",
        f"--time={resources.time}",
        f"--mem={resources.mem}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--account={SLURM_ACCOUNT}",
        f"--job-name={job_name}",
        "--signal=B:USR1@300",
        f"--output={SLURM_LOG_DIR}/{job_name}_%j.out",
        f"--error={SLURM_LOG_DIR}/{job_name}_%j.err",
    ]
    if resources.gres:
        args.append(f"--gres={resources.gres}")

    if dry_run:
        log.info("dry_run", cmd=" ".join(args))
        return 0

    r = subprocess.run(
        [*args, "--wrap", script], capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    if r.returncode != 0:
        raise RuntimeError(f"sbatch failed: {r.stderr.strip()}")

    m = re.search(r"(\d+)\s*$", r.stdout.strip())
    if not m:
        raise RuntimeError(f"Could not parse job ID from sbatch: {r.stdout.strip()}")
    job_id = int(m.group(1))
    log.info(
        "submitted",
        job_id=job_id,
        job_name=job_name,
        partition=resources.partition,
        time=resources.time,
        mem=resources.mem,
        gres=resources.gres or "none",
    )
    return job_id


def poll(
    job_id: int,
    *,
    interval: int = 60,
    max_unknown: int = 5,
    on_state: StateObserver | None = None,
) -> str:
    """Poll sacct until terminal state. Returns state string.

    *on_state(state, job_id)* is called on each state transition (optional).
    """
    unknown_count = 0
    last_state = None
    while True:
        stdout = sacct_query([job_id], "JobID,State")
        state = "UNKNOWN"
        if stdout:
            for line in stdout.strip().split("\n"):
                parts = line.strip().split("|")
                if len(parts) >= 2 and "." not in parts[0]:
                    state = parts[1].strip()
                    break

        if state != last_state:
            log.info("slurm_poll", job_id=job_id, state=state, prev=last_state or "initial")
            if on_state:
                on_state(state, job_id)
            last_state = state

        if state in _TERMINAL:
            return state
        if state == "UNKNOWN":
            unknown_count += 1
            if unknown_count > max_unknown:
                raise RuntimeError(
                    f"sacct returned UNKNOWN {max_unknown} consecutive times for job {job_id}"
                )
        else:
            unknown_count = 0
        _time.sleep(interval)


def cancel(job_id: int) -> None:
    """Cancel a SLURM job via scancel."""
    result = subprocess.run(["scancel", str(job_id)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"scancel failed for job {job_id}: {result.stderr.strip()}")
