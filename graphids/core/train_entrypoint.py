"""Shared train/test entry points for both CLI and pipeline paths.

The render -> validate -> instantiate -> trainer.<method> chain lives here
so both ``graphids.cli._training`` (dev path) and
``graphids.orchestrate.ops.entrypoint`` (pipeline path) call the same code.
"""

from __future__ import annotations

from typing import Any


def _execute(
    rendered: dict[str, Any],
    *,
    method: str = "fit",
    ckpt_path: str | None = None,
) -> None:
    """Core chain: validate -> instantiate -> Phase B exporters -> trainer.<method>."""
    from pathlib import Path

    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    from graphids.config.schemas import validate_config
    from graphids.instantiate import instantiate

    validated = validate_config(rendered)
    run = instantiate(rendered, validated=validated)

    # Phase B: wire file exporters now that run_dir is known.
    run_dir = Path(run.trainer.default_root_dir) if run.trainer.default_root_dir else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

        # Import SDK reference from __main__ (Phase A created it)
        from graphids.__main__ import _tracer_provider

        _tracer_provider.add_span_processor(SimpleSpanProcessor(
            ConsoleSpanExporter(out=open(run_dir / "traces.jsonl", "a"))  # noqa: SIM115
        ))

        # MeterProvider readers are constructor-only — create a new one
        # with the file exporter and replace the global provider.
        mp = MeterProvider(
            resource=_tracer_provider.resource,
            metric_readers=[PeriodicExportingMetricReader(
                ConsoleMetricExporter(out=open(run_dir / "metrics.jsonl", "a")),  # noqa: SIM115
                export_interval_millis=10_000,
            )],
        )
        metrics.set_meter_provider(mp)

    getattr(run.trainer, method)(run.model, datamodule=run.datamodule, ckpt_path=ckpt_path)


def run_training(
    *,
    config_path: str,
    tla: dict[str, Any] | None = None,
    overrides: list[str] | None = None,
    ckpt_path: str | None = None,
    method: str = "fit",
) -> None:
    """Dev-path: render -> validate -> instantiate -> trainer.<method>."""
    from graphids.cli.app import apply_overrides
    from graphids.config.jsonnet import render_config

    rendered = render_config(config_path, tla=tla or None)
    apply_overrides(rendered, overrides)
    if ckpt_path and "ckpt_path" not in rendered:
        rendered["ckpt_path"] = ckpt_path
    _execute(rendered, method=method, ckpt_path=ckpt_path)


def run_training_from_spec(spec: Any, method: str = "fit") -> None:
    """Pipeline-path: spec -> render -> validate -> instantiate -> trainer.<method>.

    ``spec`` is a ``TrainingSpec`` (imported lazily to avoid circular deps).
    """
    from graphids.config.jsonnet import render_config

    rendered = render_config(spec.jsonnet_path, tla=spec.jsonnet_tla or None)
    _execute(rendered, method=method)


def run_test_from_spec(spec: Any) -> None:
    """Pipeline-path shortcut for test."""
    run_training_from_spec(spec, method="test")
