"""Single indirection point for tracing + structured logging.

Every module imports from here, never from opentelemetry / structlog / logging
directly. Metrics formerly routed through OTel live in MLflow now
(``graphids/_mlflow.py``); this module owns only spans + log events.

Consumer API:
    from graphids._otel import get_logger

Lifecycle API (called by ``cli/app.py`` and ``cli/training.py``):
    from graphids._otel import init_providers, wire_file_exporters

Logging stack: call sites use ``log.info("event", key=value)``. ``structlog``'s
``render_to_log_kwargs`` processor lifts the kwargs into stdlib's ``extra=``
dict; the contrib ``LoggingHandler`` (replacement for the deprecated
``opentelemetry.sdk._logs.LoggingHandler``) bridges the resulting LogRecord
into the OTel log signal.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.instrumentation.logging.handler import LoggingHandler
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogRecordExporter,
)
from opentelemetry.sdk.resources import Resource, ResourceDetector
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

# ---------------------------------------------------------------------------
# SLURM → OTel resource attrs (inlined from deleted core/monitoring.py)
# ---------------------------------------------------------------------------

_SLURM_ENV_TO_ATTR = {
    "SLURM_JOB_ID": "slurm.job_id",
    "SLURM_JOB_PARTITION": "slurm.partition",
    "SLURM_NODELIST": "slurm.nodelist",
    "SLURM_GPUS_ON_NODE": "slurm.gpus_on_node",
    "SLURM_MEM_PER_NODE": "slurm.mem_per_node",
    "SLURM_CLUSTER_NAME": "slurm.cluster_name",
    "SLURM_JOB_NUM_NODES": "slurm.num_nodes",
    "CUDA_VISIBLE_DEVICES": "slurm.cuda_visible_devices",
}


class _SlurmResourceDetector(ResourceDetector):
    def detect(self) -> Resource:
        attrs = {
            attr: os.environ[env] for env, attr in _SLURM_ENV_TO_ATTR.items() if os.environ.get(env)
        }
        return Resource(attrs)


# ---------------------------------------------------------------------------
# Structured logging — sole consumer API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bridged to OTel via the contrib LoggingHandler.

    Call sites use ``log.info("event", key=value)`` — kwargs are lifted into
    the stdlib ``extra`` dict by ``render_to_log_kwargs``, which OTel maps to
    log-record attributes.
    """
    return structlog.get_logger(name)


def _configure_structlog() -> None:
    """Idempotent — structlog.configure is itself idempotent on repeat calls."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.stdlib.render_to_log_kwargs,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


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


def _get_providers() -> OTelProviders:
    if _providers is None:
        raise RuntimeError("OTel not initialised — call init_providers() first")
    return _providers


def init_providers(
    service_name: str = "graphids",
    *,
    wandb_entity: str = "",
    wandb_project: str = "graphids",
) -> OTelProviders:
    """Create and register OTel providers. Safe to call once per process."""
    global _providers  # noqa: PLW0603

    resource = Resource.create(
        {
            "service.name": service_name,
            **({"wandb.entity": wandb_entity} if wandb_entity else {}),
            **({"wandb.project": wandb_project} if wandb_project else {}),
        }
    ).merge(_SlurmResourceDetector().detect())

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

    lp = LoggerProvider(resource=resource)
    lp.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogRecordExporter(out=sys.stderr)))
    set_logger_provider(lp)
    logging.getLogger("graphids").addHandler(LoggingHandler(logger_provider=lp))

    _configure_structlog()

    atexit.register(lambda: (tp.shutdown(), lp.shutdown()))

    # Trampoline SIGTERM so atexit flushes before the process dies.
    # SIGUSR2 is owned by submitit's checkpoint handler (preemption
    # auto-resume — see graphids.slurm.submit._TrainingJob.checkpoint);
    # its normal exit path through submitit.core._submit lets atexit
    # fire cleanly, so we don't trampoline it. SIGKILL bypasses Python.
    import signal as _sig

    def _on_term(signum, _frame):
        sys.exit(128 + signum)

    _sig.signal(_sig.SIGTERM, _on_term)

    _providers = OTelProviders(tracer=tp, logger=lp)
    return _providers


def _jsonl_span(span) -> str:
    """One-line JSON per span for traces.jsonl (ndjson)."""
    return span.to_json(indent=None) + "\n"


def wire_file_exporters(run_dir: Path) -> None:
    """Add per-run file exporter for ``traces.jsonl``.

    Metrics no longer land on disk — MLflow captures scalar metrics and
    system telemetry (see ``graphids/_mlflow.py``). Only the span stream
    is written here, for the single ``training.fit`` span and any
    structured-log events emitted via ``_otel``.
    """
    p = _get_providers()
    run_dir.mkdir(parents=True, exist_ok=True)

    if p._file_span_processor is not None:
        p._file_span_processor.shutdown()

    # BatchSpanProcessor buffers on a background thread; the atexit-
    # registered shutdown in init_providers guarantees flush on normal
    # exit and on exception, so the master ``training.fit`` span always
    # lands in traces.jsonl.
    p._file_span_processor = BatchSpanProcessor(
        ConsoleSpanExporter(
            out=open(run_dir / "traces.jsonl", "a"),  # noqa: SIM115
            formatter=_jsonl_span,
        )
    )
    p.tracer.add_span_processor(p._file_span_processor)
