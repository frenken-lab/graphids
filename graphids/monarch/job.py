"""Monarch SlurmJob factory -- maps GraphIDS resource profiles to allocations.

Computes a combined SLURM allocation covering all pipeline stages, then
wraps it in a Monarch ``SlurmJob``. Uses ``job.state(cached_path=...)``
for the native reserve-once-iterate-many reconnection pattern.
"""

from __future__ import annotations

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


_STAGE_MODEL_TYPE: dict[str, str] = {
    "autoencoder": "vgae",
    "supervised": "gat",
}


def pipeline_job_spec(
    scale: str = "small",
    *,
    stages: list[str] | None = None,
    fusion_method: str = "bandit",
) -> MonarchJobSpec:
    """Compute a combined allocation for the requested pipeline stages.

    Only aggregates resources for stages that will actually run.
    GPU partition is used when any GPU stage (autoencoder, supervised)
    is present.  Fusion is CPU-only — including it inflates wall time
    but not partition/mem/cpus (those are dominated by GPU stages).
    """
    from graphids.slurm.resources import get_resources

    if stages is None:
        stages = ["autoencoder", "supervised", "fusion"]

    resources = []
    for stage in stages:
        if stage == "fusion":
            resources.append(get_resources(fusion_method, scale, stage))
        else:
            resources.append(get_resources(_STAGE_MODEL_TYPE[stage], scale, stage))

    total_minutes = sum(r.time_minutes for r in resources) + 30
    h, m = divmod(total_minutes, 60)

    # GPU stages drive partition and GPU count
    gpu_resources = [r for r in resources if r.gres]
    if gpu_resources:
        partition = gpu_resources[0].partition
        parts = gpu_resources[0].gres.split(":")
        gpus = int(parts[-1]) if parts[-1].isdigit() else 1
    else:
        partition = resources[0].partition
        gpus = 0

    return MonarchJobSpec(
        partition=partition,
        time=f"{h}:{m:02d}:00",
        mem=f"{max(r.mem_mb for r in resources) // 1024}G",
        cpus=max(r.cpus_per_task for r in resources),
        gpus_per_node=gpus,
    )


def chain_job_spec(
    stages: list[Any],
    *,
    job_name: str = "graphids-monarch",
) -> MonarchJobSpec:
    """Compute a combined allocation for a chain of ``StageConfig`` objects.

    Driven by ``StageConfig.resource_model`` and ``StageConfig.scale`` —
    no hardcoded model type mapping needed. Falls back to
    ``pipeline_job_spec`` logic for resource aggregation.
    """
    from graphids.slurm.resources import get_resources

    resources = []
    for cfg in stages:
        model = cfg.resource_model or cfg.model_type
        resources.append(get_resources(model, cfg.scale, cfg.stage))

    total_minutes = sum(r.time_minutes for r in resources) + 30
    h, m = divmod(total_minutes, 60)

    gpu_resources = [r for r in resources if r.gres]
    if gpu_resources:
        partition = gpu_resources[0].partition
        parts = gpu_resources[0].gres.split(":")
        gpus = int(parts[-1]) if parts[-1].isdigit() else 1
    else:
        partition = resources[0].partition
        gpus = 0

    return MonarchJobSpec(
        partition=partition,
        time=f"{h}:{m:02d}:00",
        mem=f"{max(r.mem_mb for r in resources) // 1024}G",
        cpus=max(r.cpus_per_task for r in resources),
        gpus_per_node=gpus,
        job_name=job_name,
    )


def scale_job_spec(spec: MonarchJobSpec, reason: str) -> MonarchJobSpec:
    """Inflate a MonarchJobSpec after OOM or TIMEOUT.

    Builds a temporary ``ResourceSpec``, applies ``scale_resources``,
    and maps the result back to a ``MonarchJobSpec``.
    """
    from graphids.slurm.resources import ResourceSpec, scale_resources

    gres = f"gpu:{spec.gpus_per_node}" if spec.gpus_per_node else ""
    tmp = ResourceSpec(
        partition=spec.partition,
        time=spec.time,
        mem=spec.mem,
        cpus_per_task=spec.cpus,
        num_workers=0,
        gres=gres,
    )
    scaled = scale_resources(tmp, reason)
    return MonarchJobSpec(
        partition=scaled.partition,
        time=scaled.time,
        mem=scaled.mem,
        cpus=scaled.cpus_per_task,
        gpus_per_node=spec.gpus_per_node,
        account=spec.account,
        job_name=spec.job_name,
    )


def create_slurm_job(spec: MonarchJobSpec) -> Any:
    """Create a Monarch SlurmJob from a MonarchJobSpec.

    ``python_exe`` points to ``scripts/slurm/monarch_python.sh`` — a thin
    wrapper that sources ``.env`` + CUDA config before exec'ing the venv
    Python. Monarch generates ``srun <python_exe> -c '...'``, so the
    wrapper sets up the same environment ``_preamble.sh`` provides for
    regular SLURM jobs. Without it, workers miss ``KD_GAT_LAKE_WRITE``
    and other ``.env`` vars.

    ``exclusive=True`` requests the full node, bypassing Monarch's
    ``share_node()`` → ``clusterscope`` path which can't parse OSC's
    multi-GRES sinfo output (10+ GRES types cause ``ValueError``).
    """
    from monarch.job import SlurmJob  # type: ignore[import-not-found]

    from graphids.config.constants import PROJECT_ROOT

    worker_python = str(PROJECT_ROOT / "scripts" / "slurm" / "monarch_python.sh")

    return SlurmJob(
        meshes={"pipeline": 1},
        job_name=spec.job_name,
        partition=spec.partition,
        time_limit=spec.time,
        mem=spec.mem,
        cpus_per_task=spec.cpus,
        gpus_per_node=spec.gpus_per_node,
        python_exe=worker_python,
        slurm_args=(
            f"--account={spec.account}",
            "--signal=B:USR1@300",
            "--export=ALL",
        ),
        exclusive=True,
    )
