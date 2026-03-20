"""Thin adapter: ResourceSpec -> submitit.SlurmExecutor.

The only SLURM-specific code in the project. Everything else uses
concurrent.futures.Executor (which submitit.SlurmExecutor implements).
"""

from __future__ import annotations

from enum import Enum

import submitit

from graphids.config import SLURM_ACCOUNT

from .job import ResourceSpec


class FailureCategory(Enum):
    OOM = "oom"
    TIMEOUT = "timeout"
    INFRA = "infra"
    APPLICATION = "application"


_SLURM_FAILURE_MAP = {
    "OUT_OF_MEMORY": FailureCategory.OOM,
    "TIMEOUT": FailureCategory.TIMEOUT,
    "NODE_FAIL": FailureCategory.INFRA,
    "PREEMPTED": FailureCategory.INFRA,
}


def classify_failure(job: submitit.Job) -> FailureCategory:
    """Map SLURM job state to a platform-agnostic failure category."""
    state = job.get_info().get("State", "FAILED").split()[0]
    return _SLURM_FAILURE_MAP.get(state, FailureCategory.APPLICATION)


def make_slurm_executor(
    resources: ResourceSpec,
    dep_futures: list | None = None,
    *,
    setup: list[str] | None = None,
    log_folder: str = "slurm_logs/%j",
) -> submitit.SlurmExecutor:
    """Configure a submitit SlurmExecutor from a ResourceSpec.

    dep_futures: list of submitit.Job (or any Future with .job_id).
    Extracts job IDs and passes --dependency=afterok.
    """
    executor = submitit.SlurmExecutor(folder=log_folder)

    dep_str = None
    if dep_futures:
        dep_ids = [str(f.job_id) for f in dep_futures]
        dep_str = f"afterok:{':'.join(dep_ids)}"

    executor.update_parameters(
        mem_gb=resources.memory_gb,
        gpus_per_node=resources.gpus,
        cpus_per_task=resources.cpus,
        timeout_min=int(resources.walltime.total_seconds() // 60),
        partition=resources.partition,
        account=SLURM_ACCOUNT,
        setup=setup or ["source scripts/slurm/_preamble.sh"],
        dependency=dep_str,
        exclude=resources.exclude_nodes or None,
        signal_delay_s=180,
    )
    return executor
