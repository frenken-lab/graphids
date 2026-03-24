"""Structlog + stdlib logging configuration.

Usage:
    from graphids.logging import configure_logging
    configure_logging()  # console renderer
    configure_logging(json=True, level="DEBUG")  # JSON for production
"""

from __future__ import annotations


def configure_logging(*, json: bool | None = None, level: str = "INFO") -> None:
    """One-time structlog + stdlib bridge setup."""
    import logging
    import os

    import structlog

    if json is None:
        json = os.environ.get("KD_GAT_JSON_LOGS", "").lower() in ("1", "true", "yes")

    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        foreign_pre_chain=shared,
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
