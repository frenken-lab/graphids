"""Single indirection for structured logging.

structlog renders one JSON line per event; stdlib ``logging`` distributes
it to three sync handlers:

1. ``sys.stderr`` — always on; lands in SLURM ``*_log.err``.
2. ``{GRAPHIDS_SLURM_LOG_DIR}/orchestrator_{SLURM_JOB_ID}.jsonl`` — when
   both env vars are set; restores the per-job structured log surface
   that older jobs (`orchestrator_46276348.jsonl` etc.) had and that a
   subsequent refactor lost.
3. ``{run_dir}/traces.jsonl`` — added by :func:`wire_file_exporters` once
   the run_dir is known.

All handlers are synchronous: ``log.info("event", k=v)`` writes before
the call returns. No batching, no background threads, no ``atexit``
choreography — events survive any signal kill (SIGUSR2 walltime,
SIGTERM, unhandled exceptions).

Public API (don't break — 19+ call sites):
    from graphids._otel import get_logger
    from graphids._otel import init_providers, wire_file_exporters

The previous OTel-backed implementation wired a ``TracerProvider`` plus
``BatchSpanProcessor`` for a ``training.fit`` master span that no caller
ever opened (zero ``start_as_current_span`` calls anywhere in graphids),
so ``traces.jsonl`` was always empty. Logs went through
``BatchLogRecordProcessor`` which dropped buffered events on SIGUSR2
walltime kills. Both are gone.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import structlog

_SLURM_ENV_TO_KEY = {
    "SLURM_JOB_ID": "slurm.job_id",
    "SLURM_JOB_PARTITION": "slurm.partition",
    "SLURM_NODELIST": "slurm.nodelist",
    "SLURM_GPUS_ON_NODE": "slurm.gpus_on_node",
    "SLURM_MEM_PER_NODE": "slurm.mem_per_node",
    "SLURM_CLUSTER_NAME": "slurm.cluster_name",
    "SLURM_JOB_NUM_NODES": "slurm.num_nodes",
    "CUDA_VISIBLE_DEVICES": "slurm.cuda_visible_devices",
}


def _slurm_context_processor(_logger, _name, event_dict: dict) -> dict:
    for env, key in _SLURM_ENV_TO_KEY.items():
        v = os.environ.get(env)
        if v and key not in event_dict:
            event_dict[key] = v
    return event_dict


def _build_handler(stream_or_path) -> logging.Handler:
    if isinstance(stream_or_path, Path):
        stream_or_path.parent.mkdir(parents=True, exist_ok=True)
        h: logging.Handler = logging.FileHandler(stream_or_path, encoding="utf-8")
    else:
        h = logging.StreamHandler(stream_or_path)
    h.setFormatter(logging.Formatter("%(message)s"))
    return h


_initialised = False


def init_providers(
    service_name: str = "graphids",  # noqa: ARG001
    *,
    wandb_entity: str = "",  # noqa: ARG001
    wandb_project: str = "graphids",  # noqa: ARG001
) -> None:
    """Wire structlog → stdlib ``logging.getLogger('graphids')`` with sync handlers.

    Idempotent — second call is a no-op so worker processes that re-init
    don't clobber handlers attached after the first call.
    """
    global _initialised  # noqa: PLW0603
    if _initialised:
        return

    handlers: list[logging.Handler] = [_build_handler(sys.stderr)]
    job_id = os.environ.get("SLURM_JOB_ID")
    log_dir = os.environ.get("GRAPHIDS_SLURM_LOG_DIR")
    if job_id and log_dir:
        handlers.append(_build_handler(Path(log_dir) / f"orchestrator_{job_id}.jsonl"))

    root = logging.getLogger("graphids")
    root.setLevel(logging.INFO)
    root.handlers = handlers
    root.propagate = False

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            _slurm_context_processor,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _initialised = True


def wire_file_exporters(run_dir: Path) -> None:
    """Add a per-run JSONL handler at ``{run_dir}/traces.jsonl``.

    Synchronous append-on-write: every ``log.info(...)`` call between now
    and process exit writes to this file before returning. Calling again
    with a different ``run_dir`` replaces the handler so successive fits
    don't bleed events into the prior run's file.
    """
    if not _initialised:
        init_providers()
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("graphids")
    root.handlers = [h for h in root.handlers if not getattr(h, "_run_handler", False)]
    handler = _build_handler(run_dir / "traces.jsonl")
    handler._run_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bridged to stdlib's ``graphids.<name>`` logger.

    Handlers are attached to the ``graphids`` stdlib logger, so the logger
    name must live under that tree for events to surface. Call sites use
    ``get_logger(__name__)`` which gives ``graphids.<module>`` — already
    inside the tree. External names get prefixed.
    """
    if not _initialised:
        init_providers()
    if not (name == "graphids" or name.startswith("graphids.")):
        name = f"graphids.{name}"
    return structlog.get_logger(name)
