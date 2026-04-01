"""LightningCLI subclass and kwargs. Internal — imported lazily by cli.py.

Used only by the dev path (python -m graphids fit) and validate.py.
Pipeline runs use direct instantiation via train_entrypoint.py.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.loggers import WandbLogger

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
        parser.link_arguments("data.init_args.dataset", "model.init_args.dataset")
        parser.link_arguments("data.init_args.lake_root", "model.init_args.lake_root")
        parser.link_arguments("seed_everything", "model.init_args.seed")
        parser.link_arguments("seed_everything", "data.init_args.seed")
        parser.link_arguments("model.init_args.conv_type", "data.init_args.conv_type")
        parser.link_arguments("model.init_args.heads", "data.init_args.heads")

        # Forced callbacks — registered as separate namespaces so stage YAMLs
        # that override trainer.callbacks cannot drop them (jsonargparse replaces
        # lists atomically, but these live outside the list).
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults({
            "checkpoint.monitor": "val_loss",
            "checkpoint.mode": "min",
            "checkpoint.save_top_k": 1,
            "checkpoint.save_last": True,
            "checkpoint.filename": "best_model",
        })
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({
            "early_stopping.monitor": "val_loss",
            "early_stopping.patience": 100,
            "early_stopping.mode": "min",
        })

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
