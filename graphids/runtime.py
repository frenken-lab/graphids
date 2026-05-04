"""Process-level setup — minimal. Most concerns delegated to Lightning/SLURM/env vars.

What Lightning + SLURM + env vars handle natively (NOT here, do NOT re-implement):

| Concern | Where it lives now |
|---|---|
| Preempt-resume | ``pl.plugins.environments.SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR2)`` passed to ``pl.Trainer(plugins=...)``. Calls ``scontrol requeue`` — same job ID, downstream ``afterok`` deps stay valid. |
| MLflow tracking URI | ``MLFLOW_TRACKING_URI`` env var (or ``MLFlowLogger(tracking_uri=...)``). |
| MLflow system metrics | ``MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true`` env var. |
| OMP/MKL thread pinning | SLURM script: ``export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK``. torch reads it on import. |
| MLflow run lifecycle | ``MLFlowLogger`` (lazy open + ``finalize`` on Trainer teardown). |

What only graphids cares about and Python has to set:
- structlog → JSON sync stderr with SLURM env auto-attached.
- ``multiprocessing.set_start_method("spawn")`` — CUDA + fork = corrupt context.
- ``torch.multiprocessing.set_sharing_strategy("file_system")`` — OSC vm.max_map_count
  workaround (see ``research_spawn_mmap_hpc.md`` in memory).

Idempotency via ``functools.cache`` — no manual flag bookkeeping.
"""

from __future__ import annotations

import functools
import multiprocessing
import os
from typing import Any, Literal

import structlog

_SLURM_KEYS: dict[str, str] = {
    "SLURM_JOB_ID": "slurm.job_id",
    "SLURM_CLUSTER_NAME": "slurm.cluster_name",
    "SLURM_JOB_PARTITION": "slurm.partition",
    "SLURM_NODELIST": "slurm.nodelist",
    "SLURM_CPUS_PER_TASK": "slurm.cpus_per_task",
    "SLURM_GPUS_ON_NODE": "slurm.gpus_on_node",
    "CUDA_VISIBLE_DEVICES": "slurm.cuda_visible_devices",
}


def _slurm_context(_logger: Any, _method: str, event_dict: dict) -> dict:  # noqa: ARG001
    """structlog processor: attach SLURM env to every event."""
    for env, key in _SLURM_KEYS.items():
        if (v := os.environ.get(env)) and key not in event_dict:
            event_dict[key] = v
    return event_dict


@functools.cache
def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            _slurm_context,
            structlog.processors.JSONRenderer(),
        ],
        cache_logger_on_first_use=True,
    )


@functools.cache
def _ensure_spawn() -> None:
    """spawn mp + ``file_system`` sharing. CUDA-safe + OSC-safe."""
    import torch.multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    torch.multiprocessing.set_sharing_strategy("file_system")


def setup(*, mode: Literal["compute", "ops", "render"] = "compute") -> None:
    """Idempotent. Logging always; spawn only for ``compute``.

    No tracking-URI / CPU-thread / system-metrics calls — those move to
    env vars and ``MLFlowLogger``/``SLURMEnvironment`` plugin construction
    in ``orchestrate``.
    """
    _configure_logging()
    if mode == "compute":
        _ensure_spawn()
