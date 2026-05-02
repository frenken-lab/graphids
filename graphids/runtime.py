"""Process-level setup shared by every compute path.

Idempotent. Called from :func:`graphids.orchestrate.run_row` so the direct
CLI (``graphids exec``) and the SLURM body get the same fixed environment.

Five pieces:
1. structlog → JSON-line stderr, with auto-injected SLURM context (job_id,
   partition, nodelist, cluster_name) on every event. Single sink: stderr,
   which lands in SLURM's ``*_log.err``.
2. ``spawn`` multiprocessing + ``file_system`` tensor IPC (CUDA + DataLoader
   workers are unsafe with ``fork``; see critical-constraints.md).
3. CPU-thread pinning to the SLURM allocation (PyTorch defaults intra-op
   threads to ``os.cpu_count()`` — node-wide, ignoring cgroup affinity).
4. MLflow tracking URI (delegates to :func:`graphids._mlflow.ensure_tracking_uri`).
5. SIGUSR2 preempt handler: re-submits the row with ``--ckpt-path=last.ckpt``
   and ``--dependency=afterany:$SLURM_JOB_ID``.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
from typing import Any

import structlog

_SPAWN_SET = False
_THREADS_SET = False
_LOGGING_CONFIGURED = False

# SLURM env vars that carry useful identity for log queries. Auto-attached
# to every event by the structlog processor below.
_SLURM_KEYS = {
    "SLURM_JOB_ID": "slurm.job_id",
    "SLURM_JOB_PARTITION": "slurm.partition",
    "SLURM_NODELIST": "slurm.nodelist",
    "SLURM_CLUSTER_NAME": "slurm.cluster_name",
    "SLURM_CPUS_PER_TASK": "slurm.cpus_per_task",
    "SLURM_GPUS_ON_NODE": "slurm.gpus_on_node",
    "CUDA_VISIBLE_DEVICES": "slurm.cuda_visible_devices",
}


def _slurm_context(_logger: Any, _method: str, event_dict: dict) -> dict:
    """structlog processor: attach SLURM env vars to every event."""
    for env, key in _SLURM_KEYS.items():
        if (v := os.environ.get(env)) and key not in event_dict:
            event_dict[key] = v
    return event_dict


def _configure_logging() -> None:
    """Install structlog → JSON sync stderr handler. Idempotent."""
    global _LOGGING_CONFIGURED  # noqa: PLW0603
    if _LOGGING_CONFIGURED:
        return
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
    _LOGGING_CONFIGURED = True


def _ensure_spawn() -> None:
    global _SPAWN_SET  # noqa: PLW0603
    if _SPAWN_SET:
        return
    import torch.multiprocessing  # noqa: PLC0415

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    torch.multiprocessing.set_sharing_strategy("file_system")
    _SPAWN_SET = True


def _configure_cpu_threads() -> None:
    global _THREADS_SET  # noqa: PLW0603
    if _THREADS_SET:
        return
    import torch  # noqa: PLC0415

    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    n = (int(slurm) if slurm and slurm.isdigit() else None) or os.cpu_count() or 1
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Some torch op already launched. Intra-op is still applied; that's
        # where CPU-bound ops get most of their wins.
        structlog.get_logger(__name__).warning("cpu_threads_interop_locked", intra_op=n)
    _THREADS_SET = True


def setup() -> None:
    """Idempotent process-level setup. Safe to call from anywhere."""
    from graphids._mlflow import ensure_tracking_uri  # noqa: PLC0415

    _configure_logging()
    _ensure_spawn()
    _configure_cpu_threads()
    ensure_tracking_uri()


def register_preempt_handler(row: Any) -> None:
    """SIGUSR2 → re-submit row with afterany dep on the current SLURM job.

    Pairs with ``submit_row``'s ``--signal=USR2@N`` directive. SLURM sends
    SIGUSR2 N seconds before walltime; we save no extra state (the trainer's
    ``ModelCheckpoint`` already wrote ``last.ckpt`` on the most recent epoch
    end), then re-submit ``row`` with ``ckpt_path=last.ckpt``. No-op outside
    SLURM (no ``SLURM_JOB_ID`` env var).
    """
    jid = os.environ.get("SLURM_JOB_ID")
    cluster = os.environ.get("SLURM_CLUSTER_NAME")
    if not jid or not cluster:
        return
    log = structlog.get_logger(__name__)

    def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        from graphids.slurm import submit_row  # noqa: PLC0415

        last_ckpt = f"{row.identity.run_dir}/checkpoints/last.ckpt"
        try:
            new_jid = submit_row(
                row,
                cluster=cluster,
                length="long",
                ckpt_path=last_ckpt,
                depends_on_afterany=jid,
            )
            log.info("preempt_resume_submitted", original_jid=jid, resume_jid=new_jid)
        except Exception as e:
            log.error("preempt_resume_failed", original_jid=jid, error=str(e))
        sys.exit(0)

    signal.signal(signal.SIGUSR2, _handler)
