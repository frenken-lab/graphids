"""SLURM pipeline helpers wired to GraphIDS spec envelopes."""

from __future__ import annotations

import shlex
import uuid
from pathlib import Path
from typing import Protocol

from graphids.config.constants import PHASE_MARKERS
from graphids.core.analysis.schemas import AnalysisSpec
from graphids.contracts import to_envelope
from graphids.orchestrate.contracts import TrainingSpec
from graphids.slurm.core.submit import StateObserver, cancel, poll, submit
from graphids.slurm.env import EPILOG_PATH, PREAMBLE_PATH, SLURM_LOG_DIR
from graphids.slurm.resources import ResourceSpec

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
    is_cpu = not resources.gres
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
    ]
    if is_cpu:
        lines.append("export SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1")
    lines.extend(
        [
            f"source {PREAMBLE_PATH}",
            f"_RUN_DIR={qrd}",
            f"python -m graphids from-spec --phase train --spec-file {quoted}",
            f'touch "$_RUN_DIR/{PHASE_MARKERS["train"]}"',
            "# Test/analyze are best-effort — don't kill the job on failure",
            "set +euo pipefail",
        ]
    )
    if run_test:
        lines.append(f"if python -m graphids from-spec --phase test --spec-file {quoted}; then")
        lines.append(f'  touch "$_RUN_DIR/{PHASE_MARKERS["test"]}"')
        lines.append("fi")
    if analysis_spec_file:
        aquoted = shlex.quote(str(analysis_spec_file))
        lines.append(f"if python -m graphids from-spec --phase analyze --spec-file {aquoted}; then")
        lines.append(f'  touch "$_RUN_DIR/{PHASE_MARKERS["analyze"]}"')
        lines.append("fi")
    lines.append(f'python -m graphids _finalize-record --run-dir "$_RUN_DIR"')
    lines.append(f"source {EPILOG_PATH}")
    return "\n".join(lines) + "\n"


def write_training_spec(training_spec: TrainingSpec, *, job_name: str) -> Path:
    """Persist TrainingSpec to shared filesystem for SLURM worker consumption."""
    specs_dir = Path(SLURM_LOG_DIR) / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{job_name}_{uuid.uuid4().hex}.json"
    path = specs_dir / filename
    envelope = to_envelope(training_spec, metadata={"job_name": job_name})
    path.write_text(envelope.model_dump_json())
    return path


def write_analysis_spec(analysis_spec: AnalysisSpec, *, job_name: str) -> Path:
    """Persist AnalysisSpec to shared filesystem for SLURM worker consumption."""
    specs_dir = Path(SLURM_LOG_DIR) / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{job_name}_analysis_{uuid.uuid4().hex}.json"
    path = specs_dir / filename
    envelope = to_envelope(analysis_spec, metadata={"job_name": job_name})
    path.write_text(envelope.model_dump_json())
    return path


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
            write_analysis_spec(analysis_spec, job_name=job_name) if analysis_spec else None
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
            # Spec files preserved in {SLURM_LOG_DIR}/specs/ for audit trail.
            pass

    def cancel_job(self, job_id: int) -> None:
        cancel(job_id)
