"""OpenTelemetry training callback + logger + SLURM resource detector.

Replaces ResourceProfileCallback + RunRecordCallback + DeviceStatsMonitor
(callback) and WandbLogger + CSVLogger (logger).

Callback controls span lifecycle (fit start/end, batch timing, VRAM gauges,
epoch events, cross-stage span links). Logger captures ``model.log()`` calls
from training modules as OTel histograms.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource, ResourceDetector
from opentelemetry.trace import Link, SpanContext, TraceFlags

from graphids.core.callbacks import CallbackBase

# ---------------------------------------------------------------------------
# SLURM Resource Detector
# ---------------------------------------------------------------------------

_SLURM_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_PARTITION",
    "SLURM_NODELIST",
    "SLURM_GPUS_ON_NODE",
    "SLURM_MEM_PER_NODE",
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_NUM_NODES",
    "CUDA_VISIBLE_DEVICES",
)

_ATTR_NAMES = {
    "SLURM_JOB_ID": "slurm.job_id",
    "SLURM_JOB_PARTITION": "slurm.partition",
    "SLURM_NODELIST": "slurm.nodelist",
    "SLURM_GPUS_ON_NODE": "slurm.gpus_on_node",
    "SLURM_MEM_PER_NODE": "slurm.mem_per_node",
    "SLURM_CLUSTER_NAME": "slurm.cluster_name",
    "SLURM_JOB_NUM_NODES": "slurm.num_nodes",
    "CUDA_VISIBLE_DEVICES": "slurm.cuda_visible_devices",
}


class SlurmResourceDetector(ResourceDetector):
    """Harvest SLURM environment variables into OTel resource attributes."""

    def detect(self) -> Resource:
        attrs = {}
        for env_key in _SLURM_KEYS:
            val = os.environ.get(env_key)
            if val:
                attrs[_ATTR_NAMES[env_key]] = val
        return Resource(attrs)


# ---------------------------------------------------------------------------
# OTel Training Callback
# ---------------------------------------------------------------------------


class OTelTrainingCallback(CallbackBase):
    """Per-run span + resource gauges via OpenTelemetry.

    Creates a ``training.fit`` span on fit start, records per-batch VRAM
    and timing as OTel gauges/histograms, per-epoch events with LR and
    early stopping state, and closes the span with final metrics on fit
    end (or exception). Discovers upstream stage spans and records them
    as span links for cross-stage lineage.
    """

    def __init__(
        self,
        *,
        stage: str = "",
        dataset: str = "",
        scale: str = "",
        seed: int = 0,
        model_type: str = "",
    ) -> None:
        self._span: trace.Span | None = None
        self._tracer = trace.get_tracer(__name__)
        meter = metrics.get_meter(__name__)
        self._loss_hist = meter.create_histogram("ml.train.loss", unit="1")
        self._batch_dur = meter.create_histogram("ml.batch.duration_s", unit="s")
        self._cuda_alloc = meter.create_gauge("ml.cuda.allocated_mb", unit="MiB")
        self._cuda_reserved = meter.create_gauge("ml.cuda.reserved_mb", unit="MiB")
        self._gpu_util = meter.create_gauge("ml.gpu.utilization_pct", unit="%")
        self._gpu_temp = meter.create_gauge("ml.gpu.temperature_c", unit="degC")
        self._gpu_power = meter.create_gauge("ml.gpu.power_w", unit="W")
        self._step_start: float = 0.0
        self._identity = {
            "ml.stage": stage,
            "ml.dataset": dataset,
            "ml.scale": scale,
            "ml.seed": seed,
            "ml.model_type": model_type,
        }

    # -- lifecycle ------------------------------------------------------------

    def on_fit_start(self, trainer, model: torch.nn.Module) -> None:
        # Discover upstream span links before creating the span
        links = self._discover_upstream_links(trainer)
        self._span = self._tracer.start_span("training.fit", links=links or None)
        ctx = trace.set_span_in_context(self._span)
        self._token = trace.context_api.attach(ctx)

        root_dir = trainer.default_root_dir or ""
        self._span.set_attribute("ml.run_dir", root_dir)
        self._span.set_attribute("ml.model_class", type(model).__name__)
        self._span.set_attribute("ml.max_epochs", trainer.max_epochs or 0)

        # Identity attributes from jsonnet config
        for k, v in self._identity.items():
            if v:
                self._span.set_attribute(k, v)

        # Campaign context (opt-in via env var, format "<manifest>::<cell_id>").
        # Tagging the existing span avoids a parallel status log — the
        # campaign CLI recovers cell state by querying traces.jsonl.
        raw = os.environ.get("GRAPHIDS_CAMPAIGN_CELL", "")
        manifest, _, cell_id = raw.partition("::")
        if manifest and cell_id:
            self._span.set_attribute("campaign.manifest", manifest)
            self._span.set_attribute("campaign.cell_id", cell_id)

    def on_fit_end(self, trainer, model: torch.nn.Module) -> None:
        if self._span is None:
            return
        for k, v in trainer.callback_metrics.items():
            val = float(v) if isinstance(v, (int, float, torch.Tensor)) else None
            if val is not None:
                self._span.set_attribute(f"ml.metric.{k}", round(val, 6))
        self._span.set_attribute("ml.epochs_run", trainer.current_epoch + 1)
        ckpt_cb = trainer.checkpoint_callback
        if ckpt_cb is not None and ckpt_cb.best_model_path:
            self._span.set_attribute("ml.checkpoint.best_path", str(ckpt_cb.best_model_path))
        self._span.set_status(trace.StatusCode.OK)
        self._span.end()
        trace.context_api.detach(self._token)
        self._span = None

    def on_exception(
        self,
        trainer,
        model: torch.nn.Module,
        exception: BaseException,
    ) -> None:
        if self._span is None:
            return
        self._span.set_status(trace.StatusCode.ERROR, str(exception)[:500])
        self._span.record_exception(exception)
        self._span.end()
        trace.context_api.detach(self._token)
        self._span = None

    # -- per-epoch events -----------------------------------------------------

    def on_train_epoch_end(self, trainer, model: torch.nn.Module) -> None:
        if self._span is None:
            return
        cb = trainer.callback_metrics
        attrs: dict[str, Any] = {"epoch": trainer.current_epoch}
        for key in ("train_loss", "val_loss"):
            v = cb.get(key)
            if v is not None:
                attrs[key] = round(float(v), 6)
        if trainer.optimizers:
            attrs["lr"] = trainer.optimizers[0].param_groups[0]["lr"]
        es = trainer.early_stopping_callback
        if es is not None:
            attrs["early_stopping.wait_count"] = es.wait_count
            attrs["early_stopping.best_score"] = round(float(es.best_score), 6)
        self._span.add_event("epoch.end", attributes=attrs)

    # -- per-batch metrics ----------------------------------------------------

    def on_train_batch_start(
        self,
        trainer,
        model: torch.nn.Module,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._step_start = time.monotonic()

    def on_train_batch_end(
        self,
        trainer,
        model: torch.nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        elapsed = time.monotonic() - self._step_start
        attrs = {"ml.global_step": trainer.global_step}
        self._batch_dur.record(elapsed, attrs)

        loss_val = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_val = float(outputs["loss"])
        elif isinstance(outputs, torch.Tensor):
            loss_val = float(outputs)
        if loss_val is not None:
            self._loss_hist.record(loss_val, attrs)

        # VRAM + hardware GPU stats via torch.cuda (NVML wrappers, no pynvml init needed)
        if torch.cuda.is_available():
            dev = model.device.index or 0
            self._cuda_alloc.set(torch.cuda.memory_allocated(dev) / (1024 * 1024), attrs)
            self._cuda_reserved.set(torch.cuda.memory_reserved(dev) / (1024 * 1024), attrs)
            self._gpu_util.set(torch.cuda.utilization(dev), attrs)
            self._gpu_temp.set(torch.cuda.temperature(dev), attrs)
            # TODO(verify on compute node): torch.cuda.power_draw() unit — stable docs
            # say W but NVML nvmlDeviceGetPowerUsage returns mW and historical PyTorch
            # forwards it raw. /1000.0 matches pre-swap pynvml behavior.
            self._gpu_power.set(torch.cuda.power_draw(dev) / 1000.0, attrs)

    # -- cross-stage span links -----------------------------------------------

    def _discover_upstream_links(self, trainer) -> list[Link]:
        """Read upstream traces.jsonl to create span links for KD lineage."""
        links: list[Link] = []
        dm = trainer.datamodule
        if dm is None:
            return links
        for attr in ("vgae_ckpt_path", "gat_ckpt_path"):
            ckpt_path = getattr(dm, attr, None)
            if not ckpt_path:
                continue
            # checkpoints/best_model.ckpt -> run_dir (up 2 levels)
            upstream_run_dir = Path(ckpt_path).parent.parent
            traces_file = upstream_run_dir / "traces.jsonl"
            if not traces_file.exists():
                continue
            try:
                with open(traces_file) as f:
                    for line in f:
                        span_data = json.loads(line)
                        if span_data.get("name") != "training.fit":
                            continue
                        ctx = span_data.get("context", {})
                        links.append(Link(
                            context=SpanContext(
                                trace_id=int(ctx["trace_id"], 16),
                                span_id=int(ctx["span_id"], 16),
                                is_remote=True,
                                trace_flags=TraceFlags(0x01),
                            ),
                            attributes={
                                "ml.link.stage": attr.removesuffix("_ckpt_path"),
                                "ml.link.ckpt_path": str(ckpt_path),
                            },
                        ))
                        break
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return links


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# OTel Training Logger
# ---------------------------------------------------------------------------


class OTelTrainingLogger:
    """Logger that emits ``model.log()`` calls as OTel histograms.

    Replaces WandbLogger + CSVLogger. Each unique metric name gets a
    cached histogram instrument to avoid repeated ``create_histogram``.
    """

    def __init__(self) -> None:
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
