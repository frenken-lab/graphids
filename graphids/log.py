"""Structured logging adapter for log.info("event", key=val) call sites.

Load-bearing for 89 call sites — stdlib Logger.info() raises TypeError
on arbitrary kwargs. OTel LoggingHandler (wired in __main__.py) picks up
records from this adapter automatically.
"""

from __future__ import annotations

import logging
from typing import Any


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
