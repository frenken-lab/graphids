"""LightningCLI subclass for KD-GAT pipeline training.

Bridges the project's stage-specific construction (teacher loading, curriculum
wrapping) with Lightning's Trainer-from-config grammar. The CLI handles Trainer
+ callback instantiation from config; model/DM construction stays stage-aware.

Usage from runner.py:
    result = train_stage(cfg, stage)
"""

from __future__ import annotations

import os
from typing import Any

import pytorch_lightning as pl
import structlog
import torch
from pytorch_lightning.cli import LightningCLI

log = structlog.get_logger()


class GraphIDSCLI(LightningCLI):
    """Project CLI: Trainer from config, model/DM from stage dispatch.

    Overrides ``instantiate_classes`` to inject pre-built model and datamodule
    while letting the parent handle Trainer + callback construction from config.
    """

    def __init__(self, cfg, stage: str, dm, module, **kwargs):
        self._pre_dm = dm
        self._pre_module = module
        self._cfg = cfg
        self._stage = stage
        super().__init__(
            model_class=type(module),
            datamodule_class=type(dm) if dm is not None else None,
            save_config_callback=None,
            seed_everything_default=False,  # caller handles seeding
            run=False,  # we call fit ourselves
            args=self._build_cli_args(cfg, stage),
            **kwargs,
        )

    def _build_cli_args(self, cfg, stage: str) -> dict:
        """Map resolved config to the dict structure LightningCLI expects."""
        from graphids.pipeline.callbacks import DuckDBCatalog, RunDirectorySetup

        t = cfg.training
        callbacks = [
            {"class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
             "init_args": {"dirpath": ".", "filename": "best_model",
                           "monitor": t.monitor_metric, "mode": t.monitor_mode,
                           "save_top_k": t.save_top_k, "save_on_train_epoch_end": False}},
            {"class_path": "pytorch_lightning.callbacks.EarlyStopping",
             "init_args": {"monitor": t.monitor_metric, "patience": t.patience,
                           "mode": t.monitor_mode, "check_on_train_epoch_end": False}},
        ]
        if t.device_stats:
            callbacks.append(
                {"class_path": "pytorch_lightning.callbacks.DeviceStatsMonitor",
                 "init_args": {"cpu_stats": True}})
        if t.lr_monitor:
            callbacks.append(
                {"class_path": "pytorch_lightning.callbacks.LearningRateMonitor",
                 "init_args": {"logging_interval": t.lr_monitor_interval}})
        if t.swa_enabled:
            callbacks.append(
                {"class_path": "pytorch_lightning.callbacks.StochasticWeightAveraging",
                 "init_args": {"swa_lrs": t.swa_lrs, "swa_epoch_start": t.swa_epoch_start}})
        return {
            "trainer": {
                "max_epochs": t.max_epochs,
                "accelerator": "gpu" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu",
                "devices": 1,
                "gradient_clip_val": t.gradient_clip,
                "precision": t.precision,
                "log_every_n_steps": t.log_every_n_steps,
                "accumulate_grad_batches": t.accumulate_grad_batches,
                "deterministic": t.deterministic,
                "benchmark": t.cudnn_benchmark,
                "enable_progress_bar": not bool(os.environ.get("SLURM_JOB_ID")),
                "callbacks": callbacks,
            },
        }

    def instantiate_classes(self) -> None:
        """Inject pre-built model/DM, let parent build Trainer from config."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._pre_dm
        self.model = self._pre_module
        # Add our lifecycle callbacks (need cfg/stage refs, can't be in YAML)
        from graphids.pipeline.callbacks import DuckDBCatalog, RunDirectorySetup
        trainer_config = {**self._get(self.config_init, "trainer", default={})}
        extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        extra_callbacks.extend([
            RunDirectorySetup(self._cfg, self._stage),
            DuckDBCatalog(self._cfg, self._stage),
        ])
        self.trainer = self._instantiate_trainer(trainer_config, extra_callbacks)

    def fit(self) -> dict:
        """Run trainer.fit and return results."""
        from graphids.core.models._training import gpu_cleanup

        overrides = {}
        if hasattr(self.model, "trainer_overrides"):
            overrides = self.model.trainer_overrides(self._cfg, self.datamodule)
            if "callbacks" in overrides:
                # Fusion modules replace the entire callback list
                self.trainer.callbacks = overrides.pop("callbacks")
            for k, v in overrides.items():
                setattr(self.trainer, k, v)

        self.trainer.fit(
            self.model, datamodule=self.datamodule,
            ckpt_path=os.environ.get("KD_GAT_CKPT_PATH"),
        )

        ckpt = getattr(self.trainer.checkpoint_callback, "best_model_path", "")
        metrics = {
            k: v.item() if hasattr(v, "item") else v
            for k, v in self.trainer.callback_metrics.items()
        }
        log.info("training_complete", stage=self._stage, checkpoint=ckpt)
        gpu_cleanup()
        return {"checkpoint": ckpt, "metrics": metrics}


def train_stage(cfg, stage: str) -> dict:
    """Build DM + module, train via GraphIDSCLI. Single entry point for all training stages."""
    from graphids.pipeline.stages.runner import _build_dm, _build_module

    pl.seed_everything(cfg.seed, workers=True)
    dm, device = _build_dm(cfg, stage)
    module = _build_module(cfg, stage, device, dm)
    cli = GraphIDSCLI(cfg, stage, dm, module)
    return cli.fit()
