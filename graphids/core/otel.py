"""OpenTelemetry provider lifecycle — single source for Phase A + Phase B.

Phase A (``init_providers``): TracerProvider, MeterProvider, LoggerProvider,
optional Wandb Weave OTLP, stdlib logging bridge, atexit shutdown.

Phase B (``wire_file_exporters``): per-run ``traces.jsonl`` + ``metrics.jsonl``
once ``run_dir`` is known.

Both CLI (``__main__.py``) and Monarch (``actors.py``) call the same functions.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from opentelemetry import metrics, trace
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
# Provider container
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


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------


def init_providers(
    service_name: str = "graphids",
    *,
    wandb_entity: str = "",
    wandb_project: str = "graphids",
) -> OTelProviders:
    """Create and register all OTel providers (Phase A).

    Safe to call once per process. Both ``__main__`` and Monarch actors
    call this with different ``service_name`` values.
    """
    global _providers  # noqa: PLW0603

    resource = Resource.create({
        "service.name": service_name,
        **({"wandb.entity": wandb_entity} if wandb_entity else {}),
        **({"wandb.project": wandb_project} if wandb_project else {}),
    }).merge(SlurmResourceDetector().detect())

    # TracerProvider — keep SDK reference (API type lacks add_span_processor,
    # see opentelemetry-python#3713).
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

    # MeterProvider — placeholder; Phase B replaces with file-backed provider.
    metrics.set_meter_provider(MeterProvider(resource=resource))

    # LoggerProvider — stderr + stdlib bridge.
    lp = LoggerProvider(resource=resource)
    lp.add_log_record_processor(
        BatchLogRecordProcessor(ConsoleLogRecordExporter(out=sys.stderr))
    )
    set_logger_provider(lp)
    logging.getLogger("graphids").addHandler(LoggingHandler(logger_provider=lp))

    atexit.register(lambda: (tp.shutdown(), lp.shutdown()))

    _providers = OTelProviders(tracer=tp, logger=lp)
    return _providers


# ---------------------------------------------------------------------------
# Phase B
# ---------------------------------------------------------------------------


def wire_file_exporters(run_dir: Path) -> None:
    """Add per-run file exporters for ``traces.jsonl`` and ``metrics.jsonl``.

    For multi-stage runs (Monarch), shuts down the previous stage's span
    processor so spans don't leak across ``traces.jsonl`` files.
    """
    p = get_providers()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Shut down previous stage's processor (no-op after shutdown, stays
    # registered but inert).
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
    metrics.set_meter_provider(mp)
