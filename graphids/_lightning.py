"""LightningCLI subclass and kwargs. Internal — imported lazily by cli.py.

Used only by the dev path (python -m graphids fit) and validate.py.
Pipeline runs use direct instantiation via train_entrypoint.py.

Wiring constants (link targets, callback defaults) imported from cli.py —
single source of truth shared with the pipeline path.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.loggers import WandbLogger

from graphids.cli import CHECKPOINT_DEFAULTS, EARLY_STOPPING_DEFAULTS, LINK_TARGETS
from graphids.config import CKPT_SUBPATH, WANDB_WRITE_DIR

_CKPT_DIR = str(PurePosixPath(CKPT_SUBPATH).parent)


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
