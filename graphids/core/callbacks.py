"""graphids-specific Lightning callbacks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import lightning.pytorch as pl
import torch
from structlog import get_logger

from graphids._fs import _sha256_file
from graphids._mlflow import MLflowSystemMetricsSummaryCallback

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
class SnapshotBatchProfilerCallback(pl.Callback):
    """Optional snapshot throughput instrumentation, with explicit scored work."""

    prefix: str = "profile"
    sync_cuda: bool = False
    warmup_batches: int = 5
    log_every_n_batches: int = 10
    _started: float = field(init=False, default=0.0)
    _seen: int = field(init=False, default=0)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._started = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = max(1e-9, time.monotonic() - self._started)
        self._seen += 1
        if self._seen <= self.warmup_batches or self._seen % max(
            1, self.log_every_n_batches
        ):
            return
        values = {
            "batch_seconds": elapsed,
            "scored_bins_per_second": float(batch.scored.sum()) / elapsed,
            "scored_seconds_per_second": float((batch.exposure * batch.scored).sum())
            / elapsed,
            "scored_events_per_second": float((batch.event_count * batch.scored).sum())
            / elapsed,
            "node_bin_capacity": batch.live.numel(),
            "live_node_bins": int(batch.live.sum()),
            "graph_message_slots": int(batch.edge_mask.sum()) * batch.x.shape[1],
        }
        pl_module.log_dict(
            {f"{self.prefix}/{key}": value for key, value in values.items()},
            sync_dist=False,
        )


class ResourceUtilizationCallback(MLflowSystemMetricsSummaryCallback):
    """Compatibility shim for old configs.

    Resource summaries now come from MLflow's native system metrics collector.
    ``resource/rss_gib_*`` is retained as a benchmark metric name, but it maps
    to MLflow system memory usage rather than process RSS.
    """


@dataclass
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

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
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
