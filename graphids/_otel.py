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
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from opentelemetry import metrics as _metrics
from opentelemetry import trace
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
    """Return a meter bound to our local ``MeterProvider``.

    Do NOT use the global ``_metrics.get_meter`` — it resolves against
    the first-set provider, which has no file exporter, so recordings
    go to ``/dev/null``. Our local provider is swapped per-stage by
    ``wire_file_exporters``; instruments must be created AFTER that swap.
    """
    return get_providers().meter.get_meter(name)


# Reserved attrs on stdlib LogRecord — user kwargs with these names would
# collide at emit time. LogRecord reserved names as of CPython 3.12:
_LOGRECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }
)
# Stdlib Logger._log kwargs that must pass through, not be promoted to extra:
_PASSTHROUGH = frozenset({"exc_info", "stack_info", "stacklevel", "extra"})


class _StructuredAdapter(logging.LoggerAdapter):
    """LoggerAdapter that promotes free kwargs into ``extra=`` structured fields.

    Call sites use ``log.info("event_name", key=value, ...)``; stdlib Logger
    would reject free kwargs with TypeError. This adapter collects them into
    the stdlib ``extra`` dict, which the OTel ``LoggingHandler`` maps to span
    attributes. Collisions with LogRecord reserved names are prefixed ``x_``.
    """

    def process(self, msg, kwargs):
        fields: dict = {}
        for k in list(kwargs):
            if k in _PASSTHROUGH:
                continue
            v = kwargs.pop(k)
            fields[f"x_{k}" if k in _LOGRECORD_ATTRS else k] = v
        if fields:
            merged = dict(kwargs.get("extra") or {})
            merged.update(fields)
            kwargs["extra"] = merged
        return msg, kwargs


# Public return type. Alias exists so call sites can ``from graphids._otel
# import StructuredLogger`` for annotations without importing the impl.
StructuredLogger = logging.LoggerAdapter


def get_logger(name: str) -> StructuredLogger:
    """Return a structured logger bridged to OTel via LoggingHandler.

    Supports ``log.info("event", key=value)`` — free kwargs are promoted to
    the stdlib ``extra`` dict, which OTel maps to span attributes. The
    adapter is the sole logging indirection; swap the implementation here
    to re-wire every call site.
    """
    return _StructuredAdapter(logging.getLogger(name), {})


# ---------------------------------------------------------------------------
# Provider lifecycle
# ---------------------------------------------------------------------------


@dataclass
class OTelProviders:
    """Holds SDK-level provider references (API types lack add_span_processor).

    ``meter`` is held locally — never route through the global
    ``metrics.get_meter`` because ``set_meter_provider`` is one-shot. Any
    second call (e.g. ``wire_file_exporters`` per stage) silently refuses
    and the file exporter never receives data.
    """

    tracer: TracerProvider
    meter: MeterProvider
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

    resource = Resource.create(
        {
            "service.name": service_name,
            **({"wandb.entity": wandb_entity} if wandb_entity else {}),
            **({"wandb.project": wandb_project} if wandb_project else {}),
        }
    ).merge(SlurmResourceDetector().detect())

    tp = TracerProvider(resource=resource)
    if os.environ.get("WANDB_API_KEY"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        tp.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint="https://trace.wandb.ai/otel/v1/traces",
                    headers={"wandb-api-key": os.environ["WANDB_API_KEY"]},
                )
            )
        )
    trace.set_tracer_provider(tp)

    # MeterProvider is held LOCALLY on OTelProviders. We never call
    # ``_metrics.set_meter_provider`` — see ``get_meter`` above.
    # Start with an empty-reader provider; ``wire_file_exporters``
    # replaces it with one that has a per-stage file reader.
    mp = MeterProvider(resource=resource)

    lp = LoggerProvider(resource=resource)
    lp.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogRecordExporter(out=sys.stderr)))
    set_logger_provider(lp)
    logging.getLogger("graphids").addHandler(LoggingHandler(logger_provider=lp))

    atexit.register(lambda: (tp.shutdown(), _providers.meter.shutdown(), lp.shutdown()))

    # Trampoline SLURM signals to sys.exit so the atexit registered
    # above fires before the process dies. SIGUSR1 is what
    # ``--signal=B:USR1@300`` delivers 5 minutes before wall; SIGTERM
    # covers ``scancel`` and kernel TERM. SIGKILL bypasses Python — no
    # defense.
    import signal as _sig

    def _on_term(signum, _frame):
        sys.exit(128 + signum)

    for s in (_sig.SIGUSR1, _sig.SIGTERM):
        _sig.signal(s, _on_term)

    _providers = OTelProviders(tracer=tp, meter=mp, logger=lp)
    return _providers


def _jsonl_span(span) -> str:
    """One-line JSON per span for traces.jsonl (ndjson)."""
    return span.to_json(indent=None) + "\n"


def _jsonl_metrics(data) -> str:
    """One-line JSON per flush for metrics.jsonl (ndjson).

    OTel SDK 1.40's ``MetricsData.to_json(indent=...)`` ignores the arg
    and always pretty-prints — parse and re-dump compactly so each
    export is a single NDJSON record.
    """
    return json.dumps(json.loads(data.to_json()), separators=(",", ":")) + "\n"


def wire_file_exporters(run_dir: Path) -> None:
    """Add per-run file exporters for ``traces.jsonl`` and ``metrics.jsonl``.

    Uses ``indent=None`` formatters so both files are true NDJSON (one
    record per line), directly consumable by ``polars.read_ndjson`` and
    ``duckdb.read_json_auto`` without custom splitting.
    """
    p = get_providers()
    run_dir.mkdir(parents=True, exist_ok=True)

    if p._file_span_processor is not None:
        p._file_span_processor.shutdown()

    # BatchSpanProcessor (vs SimpleSpanProcessor) buffers spans on a
    # background thread and flushes on ``shutdown()`` — paired with the
    # ``try/finally`` shutdown around ``trainer.fit()`` in ``stage.train``,
    # this means a normally-exiting or exception-raising fit always
    # flushes its spans, where the prior simple processor only flushed at
    # span-end (and so left empty traces.jsonl files when SLURM SIGTERM'd
    # mid-fit before the master ``training.fit`` span closed).
    p._file_span_processor = BatchSpanProcessor(
        ConsoleSpanExporter(
            out=open(run_dir / "traces.jsonl", "a"),  # noqa: SIM115
            formatter=_jsonl_span,
        )
    )
    p.tracer.add_span_processor(p._file_span_processor)

    # Swap in a fresh MeterProvider with the per-stage file reader. Do NOT
    # call ``_metrics.set_meter_provider`` — it's one-shot and refuses the
    # second call silently (see OTel SDK source). Instead update our local
    # handle; callbacks pick it up via ``get_providers().meter`` on next
    # instantiation (which happens per stage in the orchestrator).
    p.meter.shutdown()
    p.meter = MeterProvider(
        resource=p.tracer.resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(
                    out=open(run_dir / "metrics.jsonl", "a"),  # noqa: SIM115
                    formatter=_jsonl_metrics,
                ),
                export_interval_millis=10_000,
            )
        ],
    )
