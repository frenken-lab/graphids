"""graphids-specific Lightning callbacks."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import lightning.pytorch as pl
import torch
from structlog import get_logger

from graphids._fs import _sha256_file

log = get_logger(__name__)


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
    _train_batches_seen: int = field(init=False, default=0)
    _batch_start_s: float | None = field(init=False, default=None)
    _previous_batch_end_s: float | None = field(init=False, default=None)
    _data_wait_s: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.warmup_batches = max(0, int(self.warmup_batches))
        self.log_every_n_batches = max(1, int(self.log_every_n_batches))

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
        events_per_sec = 0.0 if batch_s <= 0.0 else events / batch_s
        metrics = {
            f"{self.prefix}/train_batch_ms": batch_s * 1000.0,
            f"{self.prefix}/train_data_wait_ms": self._data_wait_s * 1000.0,
            f"{self.prefix}/train_events_per_sec": events_per_sec,
            f"{self.prefix}/train_events": events,
        }
        for name, value in metrics.items():
            pl_module.log(name, value, on_step=True, on_epoch=False, logger=True, prog_bar=False)

        log.info(
            "temporal_batch_profile",
            prefix=self.prefix,
            train_batch=self._train_batches_seen,
            epoch=trainer.current_epoch,
            batch_idx=batch_idx,
            train_batch_ms=round(metrics[f"{self.prefix}/train_batch_ms"], 3),
            train_data_wait_ms=round(metrics[f"{self.prefix}/train_data_wait_ms"], 3),
            train_events_per_sec=round(events_per_sec, 3),
            train_events=events,
        )


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
