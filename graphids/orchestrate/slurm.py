"""SLURM helpers: sbatch submission and sacct polling.

Pure functions — no dagster, no Lightning. Used by SlurmTrainingComponent.
"""

from __future__ import annotations

import re
import subprocess
import time as _time
from pathlib import Path

import structlog

from graphids.config import PROJECT_ROOT, SLURM_ACCOUNT, SLURM_LOG_DIR
from .resources import ResourceSpec

log = structlog.get_logger()

_TERMINAL = frozenset({
    "COMPLETED", "FAILED", "OUT_OF_MEMORY", "TIMEOUT",
    "NODE_FAIL", "CANCELLED", "PREEMPTED",
})


def sacct_query(job_ids: list[str] | list[int], fmt: str,
                *, units: str = "G") -> str:
    """Run sacct and return stdout. Shared by poll() and profiler."""
    ids = ",".join(str(j) for j in job_ids)
    cmd = ["sacct", "-j", ids, "--parsable2", "--noheader",
           f"--format={fmt}", f"--units={units}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("sacct_error", stderr=r.stderr.strip())
        return ""
    return r.stdout


def generate_script(config_files: list[str], resources: ResourceSpec, *,
                    ckpt_path: Path | None = None,
                    cli_overrides: list[str] | None = None) -> str:
    """3-line sbatch script: preamble, training command, epilog."""
    parts = ["python -m graphids fit"]
    for f in config_files:
        parts.append(f"--config {Path(f).resolve()}")
    if ckpt_path and ckpt_path.exists():
        parts.append(f"--ckpt_path {ckpt_path}")
    for arg in cli_overrides or []:
        parts.append(arg)
    cmd = " ".join(parts)
    return (
        "#!/bin/bash\n"
        f"source {PROJECT_ROOT}/scripts/slurm/_preamble.sh\n"
        f"{cmd}\n"
        f"source {PROJECT_ROOT}/scripts/slurm/_epilog.sh\n"
    )


def submit(script: str, resources: ResourceSpec, *, job_name: str,
           dry_run: bool = False) -> int:
    """Submit sbatch job. Returns job ID (0 if dry_run)."""
    Path(SLURM_LOG_DIR).mkdir(parents=True, exist_ok=True)
    args = [
        "sbatch",
        f"--partition={resources.partition}", f"--time={resources.time}",
        f"--mem={resources.mem}", f"--cpus-per-task={resources.cpus_per_task}",
        f"--account={SLURM_ACCOUNT}", f"--job-name={job_name}",
        "--signal=B:USR1@300",
        f"--output={SLURM_LOG_DIR}/{job_name}_%j.out",
        f"--error={SLURM_LOG_DIR}/{job_name}_%j.err",
    ]
    if resources.gres:
        args.append(f"--gres={resources.gres}")

    if dry_run:
        log.info("dry_run", cmd=" ".join(args))
        return 0

    r = subprocess.run([*args, "--wrap", script],
                       capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"sbatch failed: {r.stderr.strip()}")

    m = re.search(r"(\d+)\s*$", r.stdout.strip())
    if not m:
        raise RuntimeError(f"Could not parse job ID from sbatch: {r.stdout.strip()}")
    job_id = int(m.group(1))
    log.info("submitted", job_id=job_id, job_name=job_name)
    return job_id


def poll(job_id: int, *, interval: int = 60, max_unknown: int = 5,
         on_state=None) -> str:
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
