"""Monarch SlurmJob factory -- maps GraphIDS resource profiles to allocations.

Computes a combined SLURM allocation covering all pipeline stages, then
wraps it in a Monarch ``SlurmJob``. Uses ``job.state(cached_path=...)``
for the native reserve-once-iterate-many reconnection pattern.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from graphids.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class MonarchJobSpec:
    """Allocation spec for a multi-stage pipeline job."""

    partition: str
    time: str
    mem: str
    cpus: int
    gpus_per_node: int = 1
    account: str = ""
    job_name: str = "graphids-monarch"

    def __post_init__(self) -> None:
        if not self.account:
            from graphids.slurm.env import SLURM_ACCOUNT

            object.__setattr__(self, "account", SLURM_ACCOUNT)


def pipeline_job_spec(
    scale: str = "small",
    *,
    fusion_method: str = "bandit",
) -> MonarchJobSpec:
    """Compute a combined allocation covering all 3 pipeline stages.

    Takes the GPU partition (needed for autoencoder + supervised),
    max CPUs/mem across stages, and sum of wall times + 30min buffer.
    """
    from graphids.slurm.resources import get_resources

    ae = get_resources("vgae", scale, "autoencoder")
    sup = get_resources("gat", scale, "supervised")
    fus = get_resources(fusion_method, scale, "fusion")

    total_minutes = ae.time_minutes + sup.time_minutes + fus.time_minutes + 30
    h, m = divmod(total_minutes, 60)

    max_mem_mb = max(ae.mem_mb, sup.mem_mb, fus.mem_mb)

    # Parse gpu count from gres string (e.g. "gpu:1" → 1)
    gpus = 1
    if ae.gres:
        parts = ae.gres.split(":")
        gpus = int(parts[-1]) if parts[-1].isdigit() else 1

    return MonarchJobSpec(
        partition=ae.partition,
        time=f"{h}:{m:02d}:00",
        mem=f"{max_mem_mb // 1024}G",
        cpus=max(ae.cpus_per_task, sup.cpus_per_task, fus.cpus_per_task),
        gpus_per_node=gpus,
    )


def create_slurm_job(spec: MonarchJobSpec) -> Any:
    """Create a Monarch SlurmJob from a MonarchJobSpec.

    Sets ``python_exe`` to this venv's interpreter so workers use
    the correct environment. Passes ``--signal=B:USR1@300`` for
    graceful pre-timeout shutdown and ``--export=ALL`` to forward
    the submission environment to workers.
    """
    from monarch.job import SlurmJob  # type: ignore[import-not-found]

    return SlurmJob(
        meshes={"pipeline": 1},
        job_name=spec.job_name,
        partition=spec.partition,
        time_limit=spec.time,
        mem=spec.mem,
        cpus_per_task=spec.cpus,
        gpus_per_node=spec.gpus_per_node,
        python_exe=sys.executable,
        slurm_args=(
            f"--account={spec.account}",
            "--signal=B:USR1@300",
            "--export=ALL",
        ),
        exclusive=False,
    )
