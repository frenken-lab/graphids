"""LightningCLI subclass and kwargs. Internal — imported lazily by cli.py.

Used only by the dev path (python -m graphids fit) and validate.py.
Pipeline runs use direct instantiation via train_entrypoint.py.

Wiring constants (link targets, callback defaults) imported from cli.py —
single source of truth shared with the pipeline path.
"""

from __future__ import annotations

import csv
import resource
import time
from pathlib import Path, PurePosixPath

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.loggers import WandbLogger

from graphids.cli import CHECKPOINT_DEFAULTS, EARLY_STOPPING_DEFAULTS, LINK_TARGETS
from graphids.config import CKPT_SUBPATH, WANDB_WRITE_DIR

_CKPT_DIR = str(PurePosixPath(CKPT_SUBPATH).parent)

_PROFILE_FIELDS = [
    "epoch", "global_step", "num_nodes", "num_edges", "num_graphs",
    "cuda_allocated_mb", "cuda_reserved_mb", "cuda_peak_mb",
    "host_rss_mb", "step_time_ms",
]


class ResourceProfileCallback(pl.Callback):
    """Per-step VRAM + batch stats → {run_dir}/resource_profile.csv.

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

        self._writer.writerow({
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
        })

    def on_fit_end(self, trainer, pl_module):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class WandbSaveConfigCallback(SaveConfigCallback):
    """Forward full jsonargparse config to wandb (Lightning #19728 workaround)."""

    def save_config(self, trainer, pl_module, stage):
        super().save_config(trainer, pl_module, stage)
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.config.update(self.config.as_dict())
                break


class GraphIDSCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        for src, tgt in LINK_TARGETS:
            parser.link_arguments(src, tgt)

        # Forced callbacks — registered as separate namespaces so stage YAMLs
        # that override trainer.callbacks cannot drop them (jsonargparse replaces
        # lists atomically, but these live outside the list).
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults(
            {f"checkpoint.{k}": v for k, v in CHECKPOINT_DEFAULTS.items()}
        )
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {f"early_stopping.{k}": v for k, v in EARLY_STOPPING_DEFAULTS.items()}
        )

        parser.add_lightning_class_args(DeviceStatsMonitor, "device_stats")

        parser.add_lightning_class_args(ResourceProfileCallback, "resource_profile")
        parser.set_defaults({"resource_profile.log_every_n_steps": 50})

    def before_instantiate_classes(self):
        """Patch parsed config: logger save_dirs + checkpoint dirpath."""
        if not self.subcommand:
            return
        subcfg = self.config[self.subcommand]
        root_dir = subcfg.trainer.default_root_dir

        # Patch logger save_dirs from config defaults/constants
        loggers = subcfg.trainer.logger
        if isinstance(loggers, list):
            for lg in loggers:
                if not hasattr(lg, "class_path"):
                    continue
                if "WandbLogger" in lg.class_path:
                    lg.init_args.save_dir = WANDB_WRITE_DIR
                elif "CSVLogger" in lg.class_path and root_dir:
                    lg.init_args.save_dir = root_dir

        # Pin checkpoint dirpath to the run directory
        if root_dir:
            subcfg.checkpoint.dirpath = f"{root_dir}/{_CKPT_DIR}"


CLI_KWARGS = dict(
    model_class=pl.LightningModule,
    datamodule_class=pl.LightningDataModule,
    subclass_mode_model=True,
    subclass_mode_data=True,
    seed_everything_default=42,
    save_config_callback=WandbSaveConfigCallback,
    save_config_kwargs={"overwrite": True},
    parser_kwargs={
        "default_env": True,
        "env_prefix": "KD_GAT",
        **{sub: {"default_config_files": ["graphids/config/defaults/trainer.yaml"]}
           for sub in ("fit", "validate", "test", "predict")},
    },
)
