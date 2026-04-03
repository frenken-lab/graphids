"""Structured logging via stdlib — drop-in replacement for structlog.

Call-site pattern is identical to the old structlog usage::

    from graphids.log import get_logger
    log = get_logger(__name__)
    log.info("event_name", key=value, key2=value2)

Configuration is done once at process startup via :func:`configure_logging`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Adapter: lets call sites pass kwargs directly (key=value) like structlog
# ---------------------------------------------------------------------------

# LogRecord attrs that are NOT user-supplied structured data
_BUILTIN_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "stacklevel", "taskName",
    "thread", "threadName",
})


class _StructuredAdapter(logging.LoggerAdapter):
    """Logger adapter that routes arbitrary kwargs into ``extra``."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra: dict[str, Any] = {**self.extra, **kwargs.pop("extra", {})}
        for k in list(kwargs):
            if k not in ("exc_info", "stack_info", "stacklevel"):
                extra[k] = kwargs.pop(k)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str | None = None) -> _StructuredAdapter:
    """Get a structured logger. Call sites use ``log.info("event", k=v)``."""
    return _StructuredAdapter(logging.getLogger(name or "graphids"), {})


# ---------------------------------------------------------------------------
# JSON formatter — produces JSONL compatible with pipeline-status --log
# ---------------------------------------------------------------------------


class _JSONFormatter(logging.Formatter):
    """One JSON object per line: {timestamp, log_level, event, ...extras}."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        obj: dict[str, Any] = {
            "timestamp": ts,
            "log_level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _BUILTIN_ATTRS and k not in obj:
                try:
                    json.dumps(v)  # only include JSON-serializable values
                    obj[k] = v
                except (TypeError, ValueError):
                    obj[k] = str(v)
        return json.dumps(obj)


# ---------------------------------------------------------------------------
# Configuration — call once at process startup
# ---------------------------------------------------------------------------

_configured = False


def configure_logging(
    *,
    jsonl_path: str | None = None,
    level: int = logging.INFO,
) -> None:
    """Configure root ``graphids`` logger.

    Args:
        jsonl_path: If set, write JSONL to this file (SLURM mode).
                    If None, write human-readable to stderr (interactive mode).
        level: Log level (default INFO).
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return
    _configured = True

    root = logging.getLogger("graphids")
    root.setLevel(level)
    root.propagate = False

    if jsonl_path:
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        handler: logging.Handler = logging.FileHandler(jsonl_path, mode="a")
        handler.setFormatter(_JSONFormatter())
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))

    root.addHandler(handler)

    # Inject SLURM job ID into all records if running under SLURM
    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job:

        class _SlurmFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                record.slurm_job_id = slurm_job  # type: ignore[attr-defined]
                return True

        root.addFilter(_SlurmFilter())
