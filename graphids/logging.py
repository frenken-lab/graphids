"""Structlog configuration. Called once at startup."""

from __future__ import annotations

import os

import structlog


def configure_logging() -> None:
    """Configure structlog. JSON mode via KD_GAT_JSON_LOGS=1."""
    json = os.environ.get("KD_GAT_JSON_LOGS", "").lower() in ("1", "true")
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
