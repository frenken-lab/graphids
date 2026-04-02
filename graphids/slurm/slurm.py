"""SLURM helpers: sbatch submission and sacct polling.

Pure functions — no dagster, no Lightning. Used by SlurmTrainingComponent.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import time as _time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import structlog

from graphids.config import PHASE_MARKERS, PROJECT_ROOT, SLURM_ACCOUNT, SLURM_LOG_DIR
from graphids.core.contracts import (
    AnalysisContract,
    AnalysisSpec,
    TrainingContract,
    TrainingSpec,
)

from .resources import ResourceSpec

log = structlog.get_logger()

StateObserver = Callable[[str, int], None]


class SlurmJobClient(Protocol):
    """Boundary for SLURM job transport used by orchestration layers."""

    def run_training_job(
        self,
        *,
        training_spec: TrainingSpec,
        resources: ResourceSpec,
        job_name: str,
        on_state: StateObserver | None = None,
        run_test: bool = True,
        analysis_spec: AnalysisSpec | None = None,
    ) -> tuple[str, int]:
        """Submit, monitor, and return (terminal_state, job_id)."""

    def cancel_job(self, job_id: int) -> None:
        """Cancel a running SLURM job."""

_TERMINAL = frozenset({
    "COMPLETED", "FAILED", "OUT_OF_MEMORY", "TIMEOUT",
    "NODE_FAIL", "CANCELLED", "PREEMPTED",
})


def sacct_query(job_ids: list[str] | list[int], fmt: str,
                *, units: str = "G", cluster: str | None = None) -> str:
    """Run sacct and return stdout. Shared by poll() and profiler."""
    ids = ",".join(str(j) for j in job_ids)
    cmd = ["sacct", "-j", ids, "--parsable2", "--noheader",
           f"--format={fmt}", f"--units={units}"]
    if cluster:
        cmd.extend(["-M", cluster])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("sacct_error", stderr=r.stderr.strip())
        return ""
    return r.stdout


def generate_script(
    resources: ResourceSpec,
    *,
    spec_file: Path,
    run_dir: str,
    run_test: bool = True,
    analysis_spec_file: Path | None = None,
) -> str:
    """Multi-command sbatch script: train, optionally test and analyze.

    Training runs under set -e (fail-fast). Test and analyze run with
    set +e so their failures don't prevent the job from reporting success
    back to the dagster orchestrator (which writes .complete markers).
    Each phase writes a marker file on success for fine-grained status.
    """
    quoted = shlex.quote(str(spec_file))
    qrd = shlex.quote(run_dir)
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"source {PROJECT_ROOT}/scripts/slurm/_preamble.sh",
        f"_RUN_DIR={qrd}",
        f"python -m graphids train-from-spec --spec-file {quoted}",
        f"touch \"$_RUN_DIR/{PHASE_MARKERS['train']}\"",
        "# Test/analyze are best-effort — don't kill the job on failure",
        "set +euo pipefail",
    ]
    if run_test:
        lines.append(f"if python -m graphids test-from-spec --spec-file {quoted}; then")
        lines.append(f"  touch \"$_RUN_DIR/{PHASE_MARKERS['test']}\"")
        lines.append("fi")
    if analysis_spec_file:
        aquoted = shlex.quote(str(analysis_spec_file))
        lines.append(f"if python -m graphids analyze-from-spec --spec-file {aquoted}; then")
        lines.append(f"  touch \"$_RUN_DIR/{PHASE_MARKERS['analyze']}\"")
        lines.append("fi")
    lines.append(f"source {PROJECT_ROOT}/scripts/slurm/_epilog.sh")
    return "\n".join(lines) + "\n"


def write_training_spec(training_spec: TrainingSpec, *, job_name: str) -> Path:
    """Persist TrainingSpec to shared filesystem for SLURM worker consumption."""
    specs_dir = Path(SLURM_LOG_DIR) / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{job_name}_{uuid.uuid4().hex}.json"
    path = specs_dir / filename
    envelope = TrainingContract.to_envelope(training_spec, metadata={"job_name": job_name})
    path.write_text(envelope.model_dump_json())
    return path


def write_analysis_spec(analysis_spec: AnalysisSpec, *, job_name: str) -> Path:
    """Persist AnalysisSpec to shared filesystem for SLURM worker consumption."""
    specs_dir = Path(SLURM_LOG_DIR) / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{job_name}_analysis_{uuid.uuid4().hex}.json"
    path = specs_dir / filename
    envelope = AnalysisContract.to_envelope(analysis_spec, metadata={"job_name": job_name})
    path.write_text(envelope.model_dump_json())
    return path


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


def cancel(job_id: int) -> None:
    """Cancel a SLURM job via scancel."""
    result = subprocess.run(["scancel", str(job_id)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"scancel failed for job {job_id}: {result.stderr.strip()}")


class SubprocessSlurmJobClient:
    """Default SLURM adapter backed by subprocess sbatch/sacct/scancel."""

    def __init__(self, *, dry_run: bool = False, poll_interval: int = 60, max_unknown: int = 5):
        self.dry_run = dry_run
        self.poll_interval = poll_interval
        self.max_unknown = max_unknown

    def run_training_job(
        self,
        *,
        training_spec: TrainingSpec,
        resources: ResourceSpec,
        job_name: str,
        on_state: StateObserver | None = None,
        run_test: bool = True,
        analysis_spec: AnalysisSpec | None = None,
    ) -> tuple[str, int]:
        """Submit training job and block until terminal state."""
        spec_file = write_training_spec(training_spec, job_name=job_name)
        analysis_spec_file = (
            write_analysis_spec(analysis_spec, job_name=job_name)
            if analysis_spec
            else None
        )
        try:
            script = generate_script(
                resources,
                spec_file=spec_file,
                run_dir=training_spec.run_dir,
                run_test=run_test,
                analysis_spec_file=analysis_spec_file,
            )
            job_id = submit(script, resources, job_name=job_name, dry_run=self.dry_run)
            if self.dry_run:
                return "DRY_RUN", 0
            state = poll(
                job_id,
                interval=self.poll_interval,
                max_unknown=self.max_unknown,
                on_state=on_state,
            )
            return state, job_id
        finally:
            if spec_file.exists():
                spec_file.unlink()
            if analysis_spec_file and analysis_spec_file.exists():
                analysis_spec_file.unlink()

    def cancel_job(self, job_id: int) -> None:
        cancel(job_id)
