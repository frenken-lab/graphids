"""Monarch SlurmJob factory — maps resource profiles to SLURM allocations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphids.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class JobSpec:
    """SLURM allocation spec for a multi-stage pipeline job."""

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

    def create_job(self) -> Any:
        """Create a Monarch SlurmJob from this spec."""
        from pathlib import Path

        from monarch.job import SlurmJob  # type: ignore[import-not-found]

        _patch_clusterscope()

        from graphids.config.constants import PROJECT_ROOT
        from graphids.slurm.env import SLURM_LOG_DIR

        log_dir = Path(SLURM_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        return SlurmJob(
            meshes={"pipeline": 1},
            job_name=self.job_name,
            partition=self.partition,
            time_limit=self.time,
            mem=self.mem,
            cpus_per_task=self.cpus,
            gpus_per_node=self.gpus_per_node,
            python_exe=str(PROJECT_ROOT / "scripts" / "slurm" / "monarch_python.sh"),
            log_dir=str(log_dir),
            slurm_args=(
                f"--account={self.account}",
                "--signal=B:USR1@300",
                "--export=ALL",
            ),
            exclusive=False,
        )


def chain_job_spec(
    stages: list[Any],
    *,
    job_name: str = "graphids-monarch",
    dataset: str | None = None,
) -> JobSpec:
    """Compute a combined allocation covering all stages in a chain."""
    from graphids.slurm.resources import get_resources

    resources = [
        get_resources(cfg.resource_model or cfg.model_type, cfg.scale, cfg.stage, dataset=dataset)
        for cfg in stages
    ]

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

    return JobSpec(
        partition=partition,
        time=f"{h}:{m:02d}:00",
        mem=f"{max(r.mem_mb for r in resources) // 1024}G",
        cpus=max(r.cpus_per_task for r in resources),
        gpus_per_node=gpus,
        job_name=job_name,
    )


def _patch_clusterscope() -> None:
    """Fix clusterscope's sinfo parsers for OSC's multi-GRES output.

    OSC nodes report 10+ comma-separated GRES types per sinfo line.
    clusterscope assumes exactly 2 fields — we patch to use rpartition.
    """
    try:
        import clusterscope.cluster_info as _cci
        import clusterscope.slurm.partition as _csp
        from clusterscope.shell import run_cli
        from clusterscope.slurm.parser import parse_gres
    except ImportError:
        return

    def _fixed_partition_resources(partition: str) -> dict:
        result = run_cli(["sinfo", "-o", "%G,%c", f"--partition={partition}", "--noheader"])
        max_gpus = max_cpus = 0
        for line in result.strip().split("\n"):
            if not line:
                continue
            gres, _, cpus = line.rpartition(",")
            max_gpus = max(max_gpus, parse_gres(gres))
            max_cpus = max(max_cpus, int(cpus.rstrip("+")))
        return {"max_gpus": max_gpus, "max_cpus": max_cpus}

    _csp.get_partition_resources = _fixed_partition_resources

    def _fixed_get_gpu(self):
        cmd = ["sinfo", "-o", "%G,%P", "--noheader"]
        if self.partition:
            cmd.extend(["-p", self.partition])
        result = run_cli(cmd)
        results, seen = [], set()
        for line in result.strip().splitlines():
            gres, _, partition = line.rpartition(",")
            partition = partition.strip("* ")
            key = gres.split("(")[0] + partition
            if key in seen:
                continue
            seen.add(key)
            parts = gres.split(":")
            if len(parts) >= 3:
                results.append(
                    _cci.GPUInfo(
                        gpu_gen=parts[1],
                        gpu_count=int(parts[2].split("(")[0]),
                        vendor="nvidia",
                        partition=partition,
                    )
                )
        if not results:
            raise RuntimeError("No GPU information found")
        return results

    _cci.SlurmClusterInfo.get_gpu_generation_and_count = _fixed_get_gpu
