"""Single indirection point for all observability.

Every module imports from here, never from opentelemetry or logging directly.
Swap the implementation by changing THIS file only.

Consumer API (19 files):
    from graphids._otel import get_tracer, get_logger

Lifecycle API (called by __main__.py and actors.py):
    from graphids._otel import init_providers, wire_file_exporters
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from opentelemetry import metrics as _metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogRecordExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from graphids.core.monitoring import SlurmResourceDetector


# ---------------------------------------------------------------------------
# Consumer API — every module uses these
# ---------------------------------------------------------------------------


def get_tracer(name: str) -> trace.Tracer:
    """Return an OTel tracer."""
    return trace.get_tracer(name)


def get_meter(name: str) -> _metrics.Meter:
    """Return an OTel meter for recording metrics."""
    return _metrics.get_meter(name)


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib logger bridged to OTel via LoggingHandler."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Provider lifecycle
# ---------------------------------------------------------------------------


@dataclass
class OTelProviders:
    """Holds SDK-level provider references (API types lack add_span_processor)."""

    tracer: TracerProvider
    logger: LoggerProvider
    _file_span_processor: SimpleSpanProcessor | None = field(default=None, repr=False)


_providers: OTelProviders | None = None


def get_providers() -> OTelProviders:
    """Return initialised providers. Raises if ``init_providers`` was not called."""
    if _providers is None:
        raise RuntimeError("OTel not initialised — call init_providers() first")
    return _providers


def init_providers(
    service_name: str = "graphids",
    *,
    wandb_entity: str = "",
    wandb_project: str = "graphids",
) -> OTelProviders:
    """Create and register all OTel providers.

    Safe to call once per process. Called from ``__main__`` on import.
    """
    global _providers  # noqa: PLW0603

    resource = Resource.create({
        "service.name": service_name,
        **({"wandb.entity": wandb_entity} if wandb_entity else {}),
        **({"wandb.project": wandb_project} if wandb_project else {}),
    }).merge(SlurmResourceDetector().detect())

    tp = TracerProvider(resource=resource)
    if os.environ.get("WANDB_API_KEY"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint="https://trace.wandb.ai/otel/v1/traces",
            headers={"wandb-api-key": os.environ["WANDB_API_KEY"]},
        )))
    trace.set_tracer_provider(tp)

    _metrics.set_meter_provider(MeterProvider(resource=resource))

    lp = LoggerProvider(resource=resource)
    lp.add_log_record_processor(
        BatchLogRecordProcessor(ConsoleLogRecordExporter(out=sys.stderr))
    )
    set_logger_provider(lp)
    logging.getLogger("graphids").addHandler(LoggingHandler(logger_provider=lp))

    atexit.register(lambda: (tp.shutdown(), lp.shutdown()))

    _providers = OTelProviders(tracer=tp, logger=lp)
    return _providers


def wire_file_exporters(run_dir: Path) -> None:
    """Add per-run file exporters for ``traces.jsonl`` and ``metrics.jsonl``."""
    p = get_providers()
    run_dir.mkdir(parents=True, exist_ok=True)

    if p._file_span_processor is not None:
        p._file_span_processor.shutdown()

    p._file_span_processor = SimpleSpanProcessor(
        ConsoleSpanExporter(out=open(run_dir / "traces.jsonl", "a"))  # noqa: SIM115
    )
    p.tracer.add_span_processor(p._file_span_processor)

    mp = MeterProvider(
        resource=p.tracer.resource,
        metric_readers=[PeriodicExportingMetricReader(
            ConsoleMetricExporter(out=open(run_dir / "metrics.jsonl", "a")),  # noqa: SIM115
            export_interval_millis=10_000,
        )],
    )
    _metrics.set_meter_provider(mp)
