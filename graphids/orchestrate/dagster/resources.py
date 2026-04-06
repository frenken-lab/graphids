"""Dagster resources for orchestration."""

from __future__ import annotations

import dagster as dg

from graphids.orchestrate.contracts import TrainingSpec
from graphids.slurm.pipeline import SlurmJobClient, SubprocessSlurmJobClient
from graphids.slurm.resources import ResourceSpec


class SlurmTrainingResource(dg.ConfigurableResource):
    """Submits training jobs to SLURM and polls for completion."""

    dry_run: bool = False
    poll_interval: int = 60
    max_unknown: int = 5

    def _client(self) -> SlurmJobClient:
        return SubprocessSlurmJobClient(
            dry_run=self.dry_run,
            poll_interval=self.poll_interval,
            max_unknown=self.max_unknown,
        )

    def submit_and_wait(
        self,
        training_spec: TrainingSpec,
        resources: ResourceSpec,
        job_name: str,
        on_state=None,
        run_test: bool = True,
        analysis_spec=None,
        dry_run: bool = False,
    ) -> tuple[str, int]:
        """Submit SLURM job and poll. Returns (state, job_id).

        ``dry_run`` at the asset level overrides the resource-level default.
        """
        client = self._client()
        if dry_run and not self.dry_run:
            client = SubprocessSlurmJobClient(
                dry_run=True,
                poll_interval=self.poll_interval,
                max_unknown=self.max_unknown,
            )
        return client.run_training_job(
            training_spec=training_spec,
            resources=resources,
            job_name=job_name,
            on_state=on_state,
            run_test=run_test,
            analysis_spec=analysis_spec,
        )
