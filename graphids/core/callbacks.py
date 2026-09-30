"""graphids-specific Lightning callbacks."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from structlog import get_logger

from graphids._fs import _sha256_file

log = get_logger(__name__)
_BYTES_PER_GIB = 1024**3


class Sha256ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """``ModelCheckpoint`` + ``<ckpt>.sha256`` sidecar after every save.

    GPFS truncation surprises happen on OSC; the sidecar is the load-time
    integrity check used by ``_fs.atomic_load``.
    """

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:  # type: ignore[override]
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            p = Path(filepath)
            p.with_suffix(p.suffix + ".sha256").write_text(_sha256_file(p) + "\n")


@dataclass
class TemporalBatchProfilerCallback(pl.Callback):
    """Log train-batch throughput and approximate input wait time."""

    prefix: str = "profile"
    warmup_batches: int = 5
    log_every_n_batches: int = 10
    sync_cuda: bool = True
    expected_train_events: int | None = None
    fail_on_event_mismatch: bool = False
    _train_batches_seen: int = field(init=False, default=0)
    _batch_start_s: float | None = field(init=False, default=None)
    _previous_batch_end_s: float | None = field(init=False, default=None)
    _data_wait_s: float = field(init=False, default=0.0)
    _event_mismatch_warned: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.warmup_batches = max(0, int(self.warmup_batches))
        self.log_every_n_batches = max(1, int(self.log_every_n_batches))
        if self.expected_train_events is not None:
            self.expected_train_events = max(1, int(self.expected_train_events))

    def _sync_cuda(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _event_count(batch: object) -> int:
        valid_mask = batch.get("valid_mask") if isinstance(batch, Mapping) else getattr(batch, "valid_mask", None)
        valid_numel = getattr(valid_mask, "sum", None)
        if callable(valid_numel):
            return int(valid_mask.bool().sum().item())
        dst = batch.get("dst") if isinstance(batch, Mapping) else getattr(batch, "dst", None)
        numel = getattr(dst, "numel", None)
        if not callable(numel):
            return 0
        return int(numel())

    @staticmethod
    def _lane_stats(batch: object) -> dict[str, float | int]:
        valid_mask = batch.get("valid_mask") if isinstance(batch, Mapping) else getattr(batch, "valid_mask", None)
        if valid_mask is None or not hasattr(valid_mask, "bool"):
            return {}
        valid = valid_mask.bool()
        if valid.ndim < 2:
            return {}
        capacity = int(valid.numel())
        events = int(valid.sum().item())
        active_lanes = int(valid.any(dim=1).sum().item())
        padding_events = capacity - events
        return {
            "train_lane_capacity_events": capacity,
            "train_active_lanes": active_lanes,
            "train_padding_events": padding_events,
            "train_padding_fraction": 0.0 if capacity <= 0 else padding_events / capacity,
        }

    def _check_expected_events(self, *, events: int, trainer: pl.Trainer, batch_idx: int) -> None:
        if self.expected_train_events is None or events == self.expected_train_events:
            return
        message = (
            f"{self.prefix}/train_events expected {self.expected_train_events} "
            f"but observed {events} at epoch={trainer.current_epoch} batch_idx={batch_idx}"
        )
        if self.fail_on_event_mismatch:
            raise RuntimeError(message)
        if not self._event_mismatch_warned:
            log.warning(
                "temporal_batch_event_mismatch",
                prefix=self.prefix,
                expected_train_events=self.expected_train_events,
                train_events=events,
                epoch=trainer.current_epoch,
                batch_idx=batch_idx,
            )
            self._event_mismatch_warned = True

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        self._sync_cuda()
        start_s = time.perf_counter()
        self._data_wait_s = (
            0.0 if self._previous_batch_end_s is None else max(0.0, start_s - self._previous_batch_end_s)
        )
        self._batch_start_s = start_s

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del outputs
        self._sync_cuda()
        end_s = time.perf_counter()
        start_s = self._batch_start_s
        self._previous_batch_end_s = end_s
        self._batch_start_s = None
        if start_s is None:
            return

        self._train_batches_seen += 1
        profiled_batches = self._train_batches_seen - self.warmup_batches
        if profiled_batches <= 0 or profiled_batches % self.log_every_n_batches != 0:
            return

        batch_s = max(0.0, end_s - start_s)
        events = self._event_count(batch)
        self._check_expected_events(events=events, trainer=trainer, batch_idx=batch_idx)
        events_per_sec = 0.0 if batch_s <= 0.0 else events / batch_s
        metrics = {
            f"{self.prefix}/train_batch_ms": batch_s * 1000.0,
            f"{self.prefix}/train_data_wait_ms": self._data_wait_s * 1000.0,
            f"{self.prefix}/train_events_per_sec": events_per_sec,
            f"{self.prefix}/train_events": events,
        }
        lane_stats = self._lane_stats(batch)
        metrics.update({f"{self.prefix}/{name}": value for name, value in lane_stats.items()})
        for name, value in metrics.items():
            pl_module.log(name, value, on_step=True, on_epoch=False, logger=True, prog_bar=False)

        log_payload = {
            "prefix": self.prefix,
            "train_batch": self._train_batches_seen,
            "epoch": trainer.current_epoch,
            "batch_idx": batch_idx,
            "train_batch_ms": round(metrics[f"{self.prefix}/train_batch_ms"], 3),
            "train_data_wait_ms": round(metrics[f"{self.prefix}/train_data_wait_ms"], 3),
            "train_events_per_sec": round(events_per_sec, 3),
            "train_events": events,
        }
        log_payload.update(
            {
                name: round(value, 3) if isinstance(value, float) else value
                for name, value in lane_stats.items()
            }
        )
        log.info(
            "temporal_batch_profile",
            **log_payload,
        )


@dataclass
class ResourceUtilizationCallback(pl.Callback):
    """Collect offline aggregate CPU, RSS, and GPU utilization metrics."""

    prefix: str = "resource"
    sample_interval_s: float = 5.0
    include_gpu: bool = True
    _samples: list[dict[str, float]] = field(init=False, default_factory=list)
    _last_sample_s: float | None = field(init=False, default=None)
    _process: Any | None = field(init=False, default=None)
    _nvml: Any | None = field(init=False, default=None)
    _gpu_handles: list[Any] = field(init=False, default_factory=list)
    _nvml_failed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.sample_interval_s = max(0.0, float(self.sample_interval_s))

    def _ensure_process(self) -> Any | None:
        if self._process is not None:
            return self._process
        try:
            import psutil
        except ImportError:
            log.warning("resource_utilization_psutil_unavailable")
            return None
        self._process = psutil.Process()
        try:
            self._process.cpu_percent(interval=None)
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail runs
            log.warning("resource_utilization_cpu_prime_failed", error=repr(exc))
        return self._process

    def _ensure_nvml(self) -> tuple[Any | None, list[Any]]:
        if not self.include_gpu or self._nvml_failed:
            return None, []
        if self._nvml is not None:
            return self._nvml, self._gpu_handles
        try:
            import pynvml

            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in range(pynvml.nvmlDeviceGetCount())]
        except Exception as exc:  # noqa: BLE001 - GPU telemetry is optional
            self._nvml_failed = True
            log.warning("resource_utilization_nvml_unavailable", error=repr(exc))
            return None, []
        self._nvml = pynvml
        self._gpu_handles = handles
        return self._nvml, self._gpu_handles

    def _sample(self, *, force: bool = False) -> None:
        now_s = time.monotonic()
        if (
            not force
            and self._last_sample_s is not None
            and now_s - self._last_sample_s < self.sample_interval_s
        ):
            return

        sample: dict[str, float] = {}
        process = self._ensure_process()
        if process is not None:
            try:
                sample[f"{self.prefix}/cpu_percent"] = float(process.cpu_percent(interval=None))
                sample[f"{self.prefix}/rss_gib"] = float(process.memory_info().rss) / _BYTES_PER_GIB
            except Exception as exc:  # noqa: BLE001 - telemetry must not fail runs
                log.warning("resource_utilization_psutil_sample_failed", error=repr(exc))

        nvml, handles = self._ensure_nvml()
        if nvml is not None and handles:
            gpu_utils: list[float] = []
            gpu_mem_used_gib: list[float] = []
            gpu_mem_free_gib: list[float] = []
            gpu_mem_total_gib: list[float] = []
            gpu_mem_percent: list[float] = []
            for handle in handles:
                try:
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                except Exception as exc:  # noqa: BLE001 - skip broken devices
                    log.warning("resource_utilization_nvml_sample_failed", error=repr(exc))
                    continue
                total = float(mem.total)
                used = float(mem.used)
                free = float(mem.free)
                gpu_utils.append(float(util.gpu))
                gpu_mem_used_gib.append(used / _BYTES_PER_GIB)
                gpu_mem_free_gib.append(free / _BYTES_PER_GIB)
                gpu_mem_total_gib.append(total / _BYTES_PER_GIB)
                gpu_mem_percent.append(0.0 if total <= 0 else used / total * 100.0)
            if gpu_utils:
                sample[f"{self.prefix}/gpu_util_percent"] = sum(gpu_utils) / len(gpu_utils)
                sample[f"{self.prefix}/gpu_util_percent_max_sample"] = max(gpu_utils)
                sample[f"{self.prefix}/gpu_mem_gib"] = max(gpu_mem_used_gib)
                sample[f"{self.prefix}/gpu_mem_free_gib"] = min(gpu_mem_free_gib)
                sample[f"{self.prefix}/gpu_mem_total_gib"] = max(gpu_mem_total_gib)
                sample[f"{self.prefix}/gpu_mem_percent"] = max(gpu_mem_percent)

        if sample:
            self._samples.append(sample)
            self._last_sample_s = now_s

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def _aggregate(self) -> dict[str, float]:
        if not self._samples:
            return {}
        metrics: dict[str, float] = {f"{self.prefix}/samples": float(len(self._samples))}
        for key in {
            metric_name
            for sample in self._samples
            for metric_name in sample
            if metric_name != f"{self.prefix}/samples"
        }:
            values = [sample[key] for sample in self._samples if key in sample]
            if not values:
                continue
            if key.endswith("_max_sample"):
                base_key = key.removesuffix("_max_sample")
                metrics[f"{base_key}_max"] = max(values)
                continue
            if key.endswith("free_gib"):
                metrics[f"{key}_min"] = min(values)
                metrics[f"{key}_mean"] = self._mean(values)
                continue
            if key.endswith("gpu_util_percent"):
                metrics[f"{key}_mean"] = self._mean(values)
                continue
            metrics[f"{key}_mean"] = self._mean(values)
            metrics[f"{key}_max"] = max(values)
        return metrics

    def _publish(self, trainer: pl.Trainer) -> None:
        self._sample(force=True)
        trainer.callback_metrics.update(self._aggregate())

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer, pl_module
        self._sample(force=True)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx
        self._sample()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        self._sample()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._publish(trainer)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer, pl_module
        self._sample(force=True)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        self._sample()

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._publish(trainer)


@dataclass
class TorchCompileCallback(pl.Callback):
    """Compile one module method in place before Lightning starts work."""

    method_name: str = "_forward_with_state"
    backend: str | None = None
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool | None = None
    fail_on_error: bool = True
    prefix: str = "compile"
    _compiled: bool = field(init=False, default=False)

    def _compile(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._compiled:
            return
        target = getattr(pl_module, self.method_name, None)
        if not callable(target):
            message = f"{type(pl_module).__name__} has no callable method {self.method_name!r}"
            if self.fail_on_error:
                raise AttributeError(message)
            log.warning("torch_compile_target_unavailable", target_method=self.method_name)
            return

        kwargs: dict[str, Any] = {
            "fullgraph": self.fullgraph,
        }
        if self.backend is not None:
            kwargs["backend"] = self.backend
        if self.mode is not None:
            kwargs["mode"] = self.mode
        if self.dynamic is not None:
            kwargs["dynamic"] = self.dynamic

        try:
            compiled = torch.compile(target, **kwargs)
        except Exception as exc:  # noqa: BLE001 - configurable diagnostic behavior
            if self.fail_on_error:
                raise
            log.warning("torch_compile_failed", target_method=self.method_name, error=repr(exc))
            trainer.callback_metrics[f"{self.prefix}/enabled"] = 0.0
            return

        setattr(pl_module, self.method_name, compiled)
        self._compiled = True
        trainer.callback_metrics[f"{self.prefix}/enabled"] = 1.0
        log.info(
            "torch_compile_enabled",
            target_method=self.method_name,
            backend=self.backend,
            mode=self.mode,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._compile(trainer, pl_module)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._compile(trainer, pl_module)


@dataclass
class VRAMDriftCallback(pl.Callback):
    """Warn-once when free VRAM shrinks past ``threshold`` across epochs.

    Budget probe captures free VRAM at build time. Over long runs the pool
    drifts (co-resident processes, PyG activation leaks). Baseline at
    fit-start, check at each epoch start. Warn-only — re-probing mid-run
    would race optimizer state; the researcher decides whether to abort.
    """

    threshold: float = 0.20
    baseline_free: int = field(init=False, default=0)
    _warned: bool = field(init=False, default=False)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if torch.cuda.is_available():
            self.baseline_free = max(1, torch.cuda.mem_get_info()[0])

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not torch.cuda.is_available() or self.baseline_free <= 1 or self._warned:
            return
        current = torch.cuda.mem_get_info()[0]
        drift = (self.baseline_free - current) / self.baseline_free
        if drift > self.threshold:
            log.warning(
                "vram_drift_detected",
                baseline_free=self.baseline_free,
                current_free=current,
                drift_frac=round(drift, 3),
                threshold=self.threshold,
                epoch=trainer.current_epoch,
            )
            self._warned = True
