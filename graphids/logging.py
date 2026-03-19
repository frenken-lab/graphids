"""Structured logging configuration (structlog + stdlib bridge).

Call ``configure_logging()`` once at process startup (cli.py:main).
All subsequent ``structlog.get_logger()`` and stdlib ``logging.getLogger()``
calls route through the shared processor pipeline.
"""

from __future__ import annotations

import logging
import os

import structlog


def configure_logging(
    *,
    json: bool | None = None,
    level: str = "INFO",
) -> None:
    """One-time logging configuration.

    Parameters
    ----------
    json:
        Force JSON output.  When *None* (default), reads
        ``KD_GAT_JSON_LOGS`` env var (truthy = JSON).
    level:
        Root log level name (``"DEBUG"``, ``"INFO"``, …).
    """
    if json is None:
        json = os.environ.get("KD_GAT_JSON_LOGS", "").lower() in ("1", "true", "yes")

    # Shared pre-processing chain (used by both structlog and stdlib paths)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # Configure structlog bound loggers (structlog.get_logger() path)
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib (Lightning, PyG, Hydra, MLflow) through the same renderer.
    # foreign_pre_chain handles records from logging.getLogger() that didn't
    # originate from structlog — they need the full processor chain.
    # Records from structlog already carry processed event_dicts.
    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
            foreign_pre_chain=shared_processors,
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
