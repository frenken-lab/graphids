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


def _default_model_type(stage: str) -> str:
    """Look up the default model_type for a stage via topology + axes."""
    from graphids.config.constants import FAMILY_FOR_MODEL_TYPE
    from graphids.config.topology import STAGE_FAMILY_MAP

    family = STAGE_FAMILY_MAP[stage]
    # Reverse lookup: first model_type that maps to this family
    for mt, fam in FAMILY_FOR_MODEL_TYPE.items():
        if fam == family:
            return mt
    raise KeyError(f"No model_type found for family {family!r}")


def pipeline_job_spec(
    scale: str = "small",
    *,
    stages: list[str] | None = None,
    fusion_method: str = "bandit",
    dataset: str | None = None,
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
            resources.append(get_resources(fusion_method, scale, stage, dataset=dataset))
        else:
            model = _default_model_type(stage)
            resources.append(get_resources(model, scale, stage, dataset=dataset))

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
    dataset: str | None = None,
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
        resources.append(get_resources(model, cfg.scale, cfg.stage, dataset=dataset))

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

    ``exclusive=False`` allows shared nodes. Requires monkey-patching
    ``clusterscope.slurm.partition.get_partition_resources`` because OSC's
    sinfo outputs 10+ comma-separated GRES types per line, but clusterscope
    assumes exactly 2 fields (``gres, cpus = line.split(",")``).
    """
    from pathlib import Path

    from monarch.job import SlurmJob  # type: ignore[import-not-found]

    _patch_clusterscope()

    from graphids.config.constants import PROJECT_ROOT
    from graphids.slurm.env import SLURM_LOG_DIR

    worker_python = str(PROJECT_ROOT / "scripts" / "slurm" / "monarch_python.sh")
    log_dir = Path(SLURM_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    return SlurmJob(
        meshes={"pipeline": 1},
        job_name=spec.job_name,
        partition=spec.partition,
        time_limit=spec.time,
        mem=spec.mem,
        cpus_per_task=spec.cpus,
        gpus_per_node=spec.gpus_per_node,
        python_exe=worker_python,
        log_dir=str(log_dir),
        slurm_args=(
            f"--account={spec.account}",
            "--signal=B:USR1@300",
            "--export=ALL",
        ),
        exclusive=False,
    )


def _patch_clusterscope() -> None:
    """Fix clusterscope's sinfo CSV parsers for OSC's multi-GRES output.

    OSC nodes report 10+ comma-separated GRES types per sinfo line.
    clusterscope assumes exactly 2 fields (``a, b = line.split(",")``).
    We patch ``partition.py`` and ``cluster_info.py`` to use
    ``rpartition(",")`` (split on last comma) instead.
    """
    try:
        import clusterscope.cluster_info as _cci
        import clusterscope.slurm.partition as _csp
        from clusterscope.shell import run_cli
        from clusterscope.slurm.parser import parse_gres
    except ImportError:
        return

    # --- partition.py: get_partition_resources ---
    def _fixed_get_partition_resources(partition: str) -> dict:
        result = run_cli(["sinfo", "-o", "%G,%c", f"--partition={partition}", "--noheader"])
        max_gpus = 0
        max_cpus = 0
        for line in result.strip().split("\n"):
            if not line:
                continue
            gres, _, cpus = line.rpartition(",")
            max_gpus = max(max_gpus, parse_gres(gres))
            max_cpus = max(max_cpus, int(cpus.rstrip("+")))
        return {"max_gpus": max_gpus, "max_cpus": max_cpus}

    _csp.get_partition_resources = _fixed_get_partition_resources

    # --- cluster_info.py: get_gpu_generation_and_count ---
    def _fixed_get_gpu(self):
        cmd = ["sinfo", "-o", "%G,%P", "--noheader"]
        if self.partition:
            cmd.extend(["-p", self.partition])
        result = run_cli(cmd)
        results = []
        seen = set()
        for line in result.strip().splitlines():
            gres, _, partition = line.rpartition(",")
            partition = partition.strip("* ")
            key = gres.split("(")[0] + partition
            if key in seen:
                continue
            seen.add(key)
            parts = gres.split(":")
            if len(parts) >= 3:
                gpu_gen = parts[1]
                gpu_count = int(parts[2].split("(")[0])
                results.append(
                    _cci.GPUInfo(
                        gpu_gen=gpu_gen, gpu_count=gpu_count, vendor="nvidia", partition=partition
                    )
                )
        if not results:
            raise RuntimeError(f"No GPU information found")
        return results

    _cci.SlurmClusterInfo.get_gpu_generation_and_count = _fixed_get_gpu
