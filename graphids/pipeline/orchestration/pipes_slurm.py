"""Dagster Pipes SLURM client — submits sbatch jobs with Pipes protocol over NFS.

Uses ``PipesFileContextInjector`` and ``PipesFileMessageReader`` for cross-node
communication via the shared NFS filesystem (no SSH, no S3).

The training process (``cli.py``) calls ``open_dagster_pipes()`` at completion to
report metrics back through the Pipes protocol.
"""

from __future__ import annotations

import time
from pathlib import Path

import structlog

import dagster as dg
from dagster._core.pipes.utils import (
    PipesFileContextInjector,
    PipesFileMessageReader,
    open_pipes_session,
)

from graphids.config import PROJECT_ROOT
from .job import ResourceSpec
from .slurm_primitives import (
    SlurmJobFailed,
    generate_sbatch_script,
    poll_until_done,
    submit_sbatch,
    write_script_file,
)

log = structlog.get_logger()

_PIPES_DIR = Path(PROJECT_ROOT) / "slurm_logs" / "pipes"
_SCRIPTS_DIR = Path(PROJECT_ROOT) / "slurm_logs"


class PipesSlurmClient(dg.PipesClient, dg.ConfigurableResource):
    """Submit SLURM jobs with Dagster Pipes protocol over shared NFS."""

    poll_interval: int = 30

    def run(
        self,
        *,
        context: dg.OpExecutionContext | dg.AssetExecutionContext,
        extras: dict | None = None,
        stage: str,
        model: str,
        scale: str,
        dataset: str,
        resources: ResourceSpec,
        seed: int | None = None,
        auxiliaries: str = "none",
        ckpt_path: str | None = None,
    ) -> dg.PipesClientCompletedInvocation:
        """Submit a SLURM job and poll until completion via Pipes protocol."""
        pipes_dir = _PIPES_DIR / f"{stage}_{model}_{scale}"
        pipes_dir.mkdir(parents=True, exist_ok=True)

        with open_pipes_session(
            context=context,
            context_injector=PipesFileContextInjector(pipes_dir / "context"),
            message_reader=PipesFileMessageReader(pipes_dir / "messages"),
            extras=extras,
        ) as session:
            # Inject Pipes env vars into the sbatch script
            pipes_env = session.get_bootstrap_env_vars()

            script = generate_sbatch_script(
                stage=stage, model=model, scale=scale, dataset=dataset,
                resources=resources, seed=seed, auxiliaries=auxiliaries,
                ckpt_path=ckpt_path, extra_env=pipes_env,
            )
            script_path = write_script_file(
                script, _SCRIPTS_DIR, model, scale, stage, auxiliaries,
            )

            job_id = submit_sbatch(script_path)
            log.info("slurm_job_submitted", job_id=job_id, model=model, scale=scale, stage=stage)

            t0 = time.monotonic()
            state, reason, node = poll_until_done(job_id, poll_interval=self.poll_interval)
            elapsed = time.monotonic() - t0

            if state != "COMPLETED":
                raise SlurmJobFailed(
                    reason=state, node=node,
                    metadata={"job_id": job_id, "elapsed_seconds": round(elapsed, 1)},
                )

            return session.get_results()


def submit_no_poll(
    stage: str,
    model: str,
    scale: str,
    dataset: str,
    resources: ResourceSpec,
    *,
    seed: int | None = None,
    auxiliaries: str = "none",
    dependency_job_id: str | None = None,
    dry_run: bool = False,
) -> str:
    """Submit a job without polling (fire-and-forget mode). Returns job ID."""
    script = generate_sbatch_script(
        stage=stage, model=model, scale=scale, dataset=dataset,
        resources=resources, seed=seed, auxiliaries=auxiliaries,
        dependency_job_id=dependency_job_id,
    )
    script_path = write_script_file(
        script, _SCRIPTS_DIR, model, scale, stage, auxiliaries,
    )
    if dry_run:
        log.info("dry_run_submit", script_path=str(script_path))
        return "dry-run"
    job_id = submit_sbatch(script_path)
    log.info("slurm_job_submitted_no_poll", job_id=job_id, model=model, scale=scale, stage=stage)
    return job_id
