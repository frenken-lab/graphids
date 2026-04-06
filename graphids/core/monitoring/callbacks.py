# Early Stopping

# Device Mononitoring

# Learning Rate Monitoring

# Model Checkpointing

#

"""Lightning callbacks owned by graphids.

Moved out of ``_lightning.py`` when Phase 3 (2026-04-05) stripped LightningCLI.
Both callbacks are plain ``pl.Callback`` subclasses — no jsonargparse, no
LightningCLI coupling. The direct instantiator in
``graphids.core.instantiate`` constructs them explicitly alongside the
standard forced callbacks (``ModelCheckpoint``, ``EarlyStopping``,
``DeviceStatsMonitor``).
"""

from __future__ import annotations

import csv
import resource
import time
from datetime import UTC
from pathlib import Path

import pytorch_lightning as pl
import torch

_PROFILE_FIELDS = [
    "epoch",
    "global_step",
    "num_nodes",
    "num_edges",
    "num_graphs",
    "cuda_allocated_mb",
    "cuda_reserved_mb",
    "cuda_peak_mb",
    "host_rss_mb",
    "step_time_ms",
]


class ResourceProfileCallback(pl.Callback):
    """Per-step VRAM + batch stats → ``{run_dir}/resource_profile.csv``.

    Logs every ``log_every_n_steps`` training steps. Overhead on non-logging
    steps is ~50ns (modulo check). Logging steps: ~0.3ms (3 CUDA calls +
    getrusage + CSV write).
    """

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every = log_every_n_steps
        self._file = None
        self._writer = None
        self._step_start: float | None = None

    def on_fit_start(self, trainer, pl_module):
        root = trainer.default_root_dir
        if root is None:
            return
        path = Path(root) / "resource_profile.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="")  # noqa: SIM115
        self._writer = csv.DictWriter(self._file, fieldnames=_PROFILE_FIELDS)
        self._writer.writeheader()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._step_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every != 0:
            return
        if self._writer is None:
            return

        step_time_ms = None
        if self._step_start is not None:
            step_time_ms = round((time.perf_counter() - self._step_start) * 1000, 1)

        cuda_allocated = cuda_reserved = cuda_peak = 0.0
        device = pl_module.device
        if device.type == "cuda":
            cuda_allocated = torch.cuda.memory_allocated(device) / 1e6
            cuda_reserved = torch.cuda.memory_reserved(device) / 1e6
            cuda_peak = torch.cuda.max_memory_allocated(device) / 1e6
            torch.cuda.reset_peak_memory_stats(device)

        host_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB→MB

        self._writer.writerow(
            {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "num_nodes": getattr(batch, "num_nodes", None),
                "num_edges": getattr(batch, "num_edges", None),
                "num_graphs": getattr(batch, "num_graphs", None),
                "cuda_allocated_mb": round(cuda_allocated, 1),
                "cuda_reserved_mb": round(cuda_reserved, 1),
                "cuda_peak_mb": round(cuda_peak, 1),
                "host_rss_mb": round(host_rss, 1),
                "step_time_ms": step_time_ms,
            }
        )

    def on_fit_end(self, trainer, pl_module):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class RunRecordCallback(pl.Callback):
    """Write structured run_record.json sidecar on fit start/end/exception.

    Captures identity fields from ``trainer.default_root_dir`` path convention
    and final metrics from ``trainer.callback_metrics``.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._record = None

    def on_fit_start(self, trainer, pl_module):
        if not self._enabled:
            return
        root = trainer.default_root_dir
        if not root:
            self._enabled = False
            return

        import os
        from datetime import datetime

        import graphids
        from graphids.core.io import parse_identity_from_run_dir, write_run_record
        from graphids.core.run_record import RunRecord

        try:
            identity = parse_identity_from_run_dir(root)
        except (IndexError, ValueError):
            self._enabled = False
            return

        self._record = RunRecord(
            status="started",
            run_dir=root,
            stage=identity["stage"],
            model_family=identity["model_family"],
            scale=identity["scale"],
            dataset=identity["dataset"],
            seed=identity["seed"],
            identity_hash=identity["identity_hash"],
            kd_tag=identity["kd_tag"],
            user=identity["user"],
            graphids_version=graphids.__version__,
            started_at=datetime.now(UTC).isoformat(),
            slurm_job_id=(
                int(os.environ["SLURM_JOB_ID"]) if "SLURM_JOB_ID" in os.environ else None
            ),
            slurm_partition=os.environ.get("SLURM_JOB_PARTITION"),
            source="dagster" if "DAGSTER_RUN_ID" in os.environ else "cli",
        )
        write_run_record(self._record, Path(root))

    def on_fit_end(self, trainer, pl_module):
        if not self._enabled or self._record is None:
            return
        self._finalize(trainer, "completed")

    def on_exception(self, trainer, pl_module, exception):
        if not self._enabled or self._record is None:
            return
        self._finalize(trainer, "failed", error=str(exception)[:500])

    def _finalize(self, trainer, status: str, error: str | None = None):
        from datetime import datetime

        from graphids.core.io import write_run_record

        metrics = {
            k: round(float(v), 6)
            for k, v in trainer.callback_metrics.items()
            if isinstance(v, (int, float, torch.Tensor))
        }
        metrics["epochs_run"] = float(trainer.current_epoch + 1)

        self._record = self._record.model_copy(
            update={
                "status": status,
                "completed_at": datetime.now(UTC).isoformat(),
                "metrics": metrics,
                "error_message": error,
            }
        )
        write_run_record(self._record, Path(self._record.run_dir))
