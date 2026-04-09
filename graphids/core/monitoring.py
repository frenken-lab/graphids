"""OpenTelemetry Lightning callback + logger.

Replaces ResourceProfileCallback + RunRecordCallback + DeviceStatsMonitor
(callback) and WandbLogger + CSVLogger (logger).

Callback controls span lifecycle (fit start/end, batch timing, VRAM gauges).
Logger captures ``self.log()`` calls from LightningModules as OTel metrics.
"""

from __future__ import annotations

import time
from typing import Any

import pytorch_lightning as pl
import torch
from opentelemetry import metrics, trace


class OTelTrainingCallback(pl.Callback):
    """Per-run span + resource gauges via OpenTelemetry.

    Creates a ``training.fit`` span on fit start, records per-batch VRAM
    and timing as OTel gauges/histograms, and closes the span with final
    metrics on fit end (or exception).
    """

    def __init__(self) -> None:
        self._span: trace.Span | None = None
        self._tracer = trace.get_tracer(__name__)
        meter = metrics.get_meter(__name__)
        self._loss_hist = meter.create_histogram("ml.train.loss", unit="1")
        self._batch_dur = meter.create_histogram("ml.batch.duration_s", unit="s")
        self._cuda_alloc = meter.create_gauge("ml.cuda.allocated_mb", unit="MiB")
        self._cuda_reserved = meter.create_gauge("ml.cuda.reserved_mb", unit="MiB")
        self._step_start: float = 0.0

    # -- lifecycle ------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._span = self._tracer.start_span("training.fit")
        ctx = trace.set_span_in_context(self._span)
        # Attach as current so child spans (epochs, batches) nest correctly
        self._token = trace.context_api.attach(ctx)
        root_dir = trainer.default_root_dir or ""
        self._span.set_attribute("ml.run_dir", root_dir)
        self._span.set_attribute("ml.model_class", type(pl_module).__name__)
        self._span.set_attribute("ml.max_epochs", trainer.max_epochs or 0)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._span is None:
            return
        # Record final callback metrics as span attributes
        for k, v in trainer.callback_metrics.items():
            val = float(v) if isinstance(v, (int, float, torch.Tensor)) else None
            if val is not None:
                self._span.set_attribute(f"ml.metric.{k}", round(val, 6))
        self._span.set_attribute("ml.epochs_run", trainer.current_epoch + 1)
        self._span.set_status(trace.StatusCode.OK)
        self._span.end()
        trace.context_api.detach(self._token)
        self._span = None

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        if self._span is None:
            return
        self._span.set_status(trace.StatusCode.ERROR, str(exception)[:500])
        self._span.record_exception(exception)
        self._span.end()
        trace.context_api.detach(self._token)
        self._span = None

    # -- per-batch metrics ----------------------------------------------------

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._step_start = time.monotonic()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        elapsed = time.monotonic() - self._step_start
        attrs = {"ml.global_step": trainer.global_step}
        self._batch_dur.record(elapsed, attrs)

        # Loss from outputs (Lightning returns dict or tensor)
        loss_val = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_val = float(outputs["loss"])
        elif isinstance(outputs, torch.Tensor):
            loss_val = float(outputs)
        if loss_val is not None:
            self._loss_hist.record(loss_val, attrs)

        # VRAM gauges (only when CUDA is available)
        if torch.cuda.is_available():
            self._cuda_alloc.set(
                torch.cuda.memory_allocated() / (1024 * 1024), attrs
            )
            self._cuda_reserved.set(
                torch.cuda.memory_reserved() / (1024 * 1024), attrs
            )


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict for span attributes (OTel requires flat keys)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _coerce(v: Any) -> str | int | float | bool:
    """Coerce a value to an OTel-compatible attribute type."""
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


class OTelTrainingLogger(pl.loggers.Logger):
    """Lightning Logger that emits ``self.log()`` calls as OTel histograms.

    Replaces WandbLogger + CSVLogger. Each unique metric name gets a
    cached histogram instrument to avoid repeated ``create_histogram``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._meter = metrics.get_meter(__name__)
        self._instruments: dict[str, metrics.Histogram] = {}

    @property
    def name(self) -> str:
        return "otel"

    @property
    def version(self) -> str | int | None:
        return None

    def log_metrics(self, metrics_dict: dict[str, float], step: int | None = None) -> None:
        attrs = {"step": step} if step is not None else {}
        for name, value in metrics_dict.items():
            if name not in self._instruments:
                self._instruments[name] = self._meter.create_histogram(name)
            self._instruments[name].record(value, attrs)

    def log_hyperparams(self, params: dict[str, Any] | Any, *args: Any, **kwargs: Any) -> None:
        if not isinstance(params, dict):
            return
        span = trace.get_current_span()
        for k, v in _flatten(params).items():
            span.set_attribute(f"hparam.{k}", _coerce(v))
