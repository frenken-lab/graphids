"""CLI entry point: python -m graphids <subcommand>

Training:
  fit / test / validate / predict  — Lightning Trainer methods on stage configs

Analysis:
  analyze                          — generate analysis artifacts from checkpoints

Data:
  rebuild-caches                   — rebuild preprocessed graph caches
  stage-data                       — NFS -> scratch -> TMPDIR staging
  extract-fusion-states            — extract VGAE+GAT latent states for fusion

Orchestration:
  monarch-run / monarch-sweep      — run pipeline via Monarch actors
  pipeline-status                  — aggregated status from DuckDB catalog
  rebuild-catalog                  — rebuild DuckDB from traces.jsonl span data

SLURM:
  submit-profile <job>             — print resource profile for submit.sh
  probe-budget                     — hardware cost model measurement
"""

from __future__ import annotations

import atexit
import logging
import os
import sys

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogRecordExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Phase A: providers, wandb OTLP, logging bridge.
# File exporters (Phase B) are added in train_entrypoint._execute()
# once run_dir is known.

resource = Resource.create({
    "service.name": "graphids",
    "slurm.job_id": os.environ.get("SLURM_JOB_ID", ""),
    "wandb.entity": os.environ.get("WANDB_ENTITY", ""),
    "wandb.project": os.environ.get("WANDB_PROJECT", "graphids"),
})

# Keep SDK references — get_tracer_provider() returns the API type which
# lacks add_span_processor (opentelemetry-python#3713).
_tracer_provider = TracerProvider(resource=resource)
if os.environ.get("WANDB_API_KEY"):
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
        endpoint="https://trace.wandb.ai/otel/v1/traces",
        headers={"wandb-api-key": os.environ["WANDB_API_KEY"]},
    )))
trace.set_tracer_provider(_tracer_provider)

# MeterProvider readers are constructor-only. Phase B replaces this with
# a file-backed provider once run_dir is known.
metrics.set_meter_provider(MeterProvider(resource=resource))

_logger_provider = LoggerProvider(resource=resource)
_logger_provider.add_log_record_processor(
    BatchLogRecordProcessor(ConsoleLogRecordExporter(out=sys.stderr))
)
set_logger_provider(_logger_provider)
logging.getLogger("graphids").addHandler(LoggingHandler(logger_provider=_logger_provider))

atexit.register(lambda: (_tracer_provider.shutdown(), _logger_provider.shutdown()))

# Register command modules (each decorates app with @app.command)
import graphids.cli._analysis  # noqa: E402, F401
import graphids.cli._data  # noqa: E402, F401
import graphids.cli._monarch  # noqa: E402, F401
import graphids.cli._orchestrate  # noqa: E402, F401
import graphids.cli._slurm  # noqa: E402, F401
import graphids.cli._training  # noqa: E402, F401
from graphids.cli.app import app  # noqa: E402


def main() -> None:
    app()


if __name__ == "__main__":
    main()
