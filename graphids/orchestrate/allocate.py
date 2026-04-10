"""SLURM allocation primitives — independent of pipeline semantics.

``build_slurm_job(spec)`` creates a Monarch ``SlurmJob`` from a pure
``JobSpec``; ``spawn_actor(job, ...)`` spawns a ``PipelineActor`` on
the job's mesh. Neither knows anything about stages, chains, or
datasets — that lives in ``chain.py`` / ``run.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphids._otel import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class JobSpec:
    """SLURM allocation spec — no pipeline knowledge."""

    partition: str
    time: str
    mem: str
    cpus: int
    gpus_per_node: int = 1
    account: str = ""
    job_name: str = "graphids-monarch"

    def __post_init__(self) -> None:
        if not self.account:
            from graphids._slurm import slurm_account

            object.__setattr__(self, "account", slurm_account())


def build_slurm_job(spec: JobSpec) -> Any:
    """Create a Monarch ``SlurmJob`` from ``spec``.

    Patches clusterscope for OSC's SLURM config and ensures the log
    dir exists before handing off to Monarch.
    """
    from monarch.job import SlurmJob  # type: ignore[import-not-found]

    from graphids._slurm import patch_clusterscope_for_osc, slurm_log_dir
    from graphids.config.constants import PROJECT_ROOT

    patch_clusterscope_for_osc()

    log_dir = Path(slurm_log_dir())
    log_dir.mkdir(parents=True, exist_ok=True)

    return SlurmJob(
        meshes={"pipeline": 1},
        job_name=spec.job_name,
        partition=spec.partition,
        time_limit=spec.time,
        mem=spec.mem,
        cpus_per_task=spec.cpus,
        gpus_per_node=spec.gpus_per_node,
        python_exe=str(PROJECT_ROOT / "scripts" / "slurm" / "monarch_python.sh"),
        log_dir=str(log_dir),
        slurm_args=(
            f"--account={spec.account}",
            "--signal=B:USR1@300",
            "--export=ALL",
        ),
        exclusive=False,
    )


def spawn_actor(job: Any, *, gpus_per_node: int, lake_root: str) -> Any:
    """Spawn a ``PipelineActor`` on ``job``'s pipeline mesh."""
    from graphids.orchestrate.actors import PipelineActor

    proc_mesh = job.state().pipeline.spawn_procs(per_host={"gpus": gpus_per_node})
    return proc_mesh.spawn("pipeline", PipelineActor, lake_root=lake_root)


def configure_monarch() -> None:
    """Apply Monarch process-lifecycle settings for pipeline runs.

    Called once before ``build_slurm_job`` in a driver. Keeps Monarch
    config out of ``run_pipeline`` so tests can patch it.
    """
    from monarch.config import configure  # type: ignore[import-not-found]

    configure(
        enable_log_forwarding=True,
        process_exit_timeout="60s",
        cleanup_timeout="30s",
        mesh_terminate_timeout="30s",
        host_spawn_ready_timeout="120s",
    )
