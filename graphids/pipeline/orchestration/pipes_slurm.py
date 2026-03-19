"""Dagster Pipes SLURM client — thin wrapper over shared slurm_client primitives.

Adds Dagster-specific concerns: artifact validation via Pydantic contracts,
Lightning checkpoint discovery for TIMEOUT resume, and script file management.

Generic SLURM primitives (sbatch gen, sacct, retry scaling) live in
``slurm_client.py`` and are re-exported here for backward compatibility.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from graphids.config import PROJECT_ROOT

from .job import ResourceSpec
from .slurm_client import (
    FAILURE_REACTIONS,
    RESOURCE_PROFILES,
    SlurmJobFailed,
    generate_sbatch_script,
    get_resources,
    poll_until_done,
    scale_resources,
    submit_sbatch,
)

# Re-export for backward compatibility (dagster_defs.py, tests, __init__.py)
__all__ = [
    "FAILURE_REACTIONS",
    "RESOURCE_PROFILES",
    "PipesSlurmClient",
    "SlurmJobFailed",
    "generate_sbatch_script",
    "get_resources",
    "scale_resources",
]

log = logging.getLogger(__name__)


class PipesSlurmClient:
    """Submit sbatch jobs and poll via sacct for Dagster orchestration.

    In dry_run mode, logs the generated script without submitting.
    """

    def __init__(
        self,
        project_root: str | None = None,
        poll_interval: int = 30,
        dry_run: bool = False,
    ):
        self.project_root = project_root or str(PROJECT_ROOT)
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self._scripts_dir = Path(self.project_root) / "slurm_logs"

    def _write_script(
        self,
        stage: str,
        model: str,
        scale: str,
        dataset: str,
        resources: ResourceSpec,
        *,
        seed: int | None = None,
        auxiliaries: str = "none",
        ckpt_path: str | None = None,
        dependency_job_id: str | None = None,
    ) -> tuple[str, Path]:
        """Generate sbatch script, write to disk, return (script_content, path)."""
        script = generate_sbatch_script(
            stage=stage,
            model=model,
            scale=scale,
            dataset=dataset,
            resources=resources,
            seed=seed,
            auxiliaries=auxiliaries,
            ckpt_path=ckpt_path,
            dependency_job_id=dependency_job_id,
            project_root=self.project_root,
        )

        aux_tag = f"_{auxiliaries}" if auxiliaries != "none" else ""
        script_name = f"dagster_{model}_{scale}_{stage}{aux_tag}.sbatch"
        script_path = self._scripts_dir / script_name
        self._scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        log.info("Wrote sbatch script: %s", script_path)
        return script, script_path

    def run(
        self,
        stage: str,
        model: str,
        scale: str,
        dataset: str,
        resources: ResourceSpec,
        *,
        seed: int | None = None,
        auxiliaries: str = "none",
        ckpt_path: str | None = None,
        dependency_job_id: str | None = None,
    ) -> dict:
        """Submit a SLURM job and poll until completion.

        Returns a metadata dict with job_id, state, elapsed, etc.
        Raises SlurmJobFailed on terminal failure states.
        """
        script, script_path = self._write_script(
            stage, model, scale, dataset, resources,
            seed=seed, auxiliaries=auxiliaries,
            ckpt_path=ckpt_path, dependency_job_id=dependency_job_id,
        )

        if self.dry_run:
            log.info("[DRY RUN] Would submit:\n%s", script)
            return {"job_id": "dry-run", "state": "DRY_RUN", "script_path": str(script_path)}

        job_id = submit_sbatch(script_path, cwd=self.project_root)
        log.info("Submitted SLURM job %s for %s/%s/%s", job_id, model, scale, stage)

        t0 = time.monotonic()
        state, reason, node = poll_until_done(job_id, poll_interval=self.poll_interval)
        elapsed = time.monotonic() - t0

        metadata = {
            "job_id": job_id,
            "state": state,
            "reason": reason,
            "node": node,
            "elapsed_seconds": round(elapsed, 1),
            "script_path": str(script_path),
        }

        if state == "COMPLETED":
            artifacts_ok = self._validate_artifacts(dataset, model, scale, stage, auxiliaries)
            metadata["artifacts_valid"] = artifacts_ok
            if not artifacts_ok:
                log.warning(
                    "Job %s COMPLETED but artifacts missing for %s/%s/%s",
                    job_id, model, scale, stage,
                )
            return metadata

        # Terminal failure
        ckpt = self._find_checkpoint(dataset, model, scale, stage, auxiliaries)
        raise SlurmJobFailed(reason=state, node=node, ckpt_path=ckpt, metadata=metadata)

    def submit_no_poll(
        self,
        stage: str,
        model: str,
        scale: str,
        dataset: str,
        resources: ResourceSpec,
        *,
        seed: int | None = None,
        auxiliaries: str = "none",
        dependency_job_id: str | None = None,
    ) -> str:
        """Submit a job without polling. Returns job ID.

        Used for fire-and-forget mode with ``--dependency=afterok`` chains.
        """
        _script, script_path = self._write_script(
            stage, model, scale, dataset, resources,
            seed=seed, auxiliaries=auxiliaries, dependency_job_id=dependency_job_id,
        )

        if self.dry_run:
            log.info("[DRY RUN] Would submit: %s", script_path)
            return "dry-run"

        job_id = submit_sbatch(script_path, cwd=self.project_root)
        log.info("Submitted (no-poll) SLURM job %s for %s/%s/%s", job_id, model, scale, stage)
        return job_id

    # ----- Dagster-specific helpers -----

    def _validate_artifacts(
        self,
        dataset: str,
        model: str,
        scale: str,
        stage: str,
        auxiliaries: str,
        seed: int = 42,
    ) -> bool:
        """Check that expected artifacts exist after stage completion."""
        from pydantic import ValidationError

        from graphids.config import EvaluationArtifact, TrainingArtifact, lake_run_dir

        if stage == "preprocess":
            return True

        aux = auxiliaries if auxiliaries != "none" else ""
        stage_path = lake_run_dir(
            lake_root=os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns"),
            dataset=dataset, model_type=model, scale=scale,
            stage=stage, aux=aux, seed=seed, production=True,
        )

        if not stage_path.exists():
            log.warning("Artifact validation failed for %s (seed=%d)", stage_path, seed)
            return False

        contract = EvaluationArtifact if stage == "evaluation" else TrainingArtifact
        try:
            contract.from_stage_dir(stage_path)
            return True
        except (ValidationError, ValueError):
            log.warning("Artifact validation failed for %s (seed=%d)", stage_path, seed)
            return False

    def _find_checkpoint(
        self,
        dataset: str,
        model: str,
        scale: str,
        stage: str,
        auxiliaries: str,
        seed: int = 42,
    ) -> str | None:
        """Look for a Lightning auto-save checkpoint after TIMEOUT."""
        from graphids.config import lake_run_dir

        aux = auxiliaries if auxiliaries != "none" else ""
        stage_path = lake_run_dir(
            lake_root=os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns"),
            dataset=dataset, model_type=model, scale=scale,
            stage=stage, aux=aux, seed=seed, production=True,
        )
        ckpt = stage_path / ".pl_auto_save.ckpt"
        if ckpt.exists():
            return str(ckpt)
        return None
