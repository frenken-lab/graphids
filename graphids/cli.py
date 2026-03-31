"""Shared LightningCLI subclass — single definition used by __main__ and orchestrate."""
from __future__ import annotations

from pathlib import PurePosixPath

import pytorch_lightning as pl
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

    def before_instantiate_classes(self):
        """Patch parsed config: logger save_dirs + checkpoint dirpath."""
        if not self.subcommand:
            return
        subcfg = self.config[self.subcommand]
        root_dir = subcfg.trainer.default_root_dir

        # Patch logger save_dirs from write_paths.yaml constants
        loggers = subcfg.trainer.logger
        if isinstance(loggers, list):
            for lg in loggers:
                if not hasattr(lg, "class_path"):
                    continue
                if "WandbLogger" in lg.class_path:
                    lg.init_args.save_dir = WANDB_WRITE_DIR
                elif "CSVLogger" in lg.class_path and root_dir:
                    lg.init_args.save_dir = root_dir

        if not root_dir:
            return

        # Ensure ModelCheckpoint is always present and dirpath is pinned.
        # Stage YAMLs that override trainer.callbacks can accidentally drop it
        # (jsonargparse replaces lists, not merges), so cli.py is the safety net.
        cbs = subcfg.trainer.callbacks
        if not isinstance(cbs, list):
            cbs = []
            subcfg.trainer.callbacks = cbs

        has_ckpt = any(
            hasattr(cb, "class_path") and "ModelCheckpoint" in cb.class_path
            for cb in cbs
        )
        if not has_ckpt:
            from jsonargparse import Namespace

            cbs.append(Namespace(
                class_path="pytorch_lightning.callbacks.ModelCheckpoint",
                init_args=Namespace(
                    monitor="val_loss", mode="min", save_top_k=1,
                    save_last=True, filename="best_model",
                    dirpath=f"{root_dir}/{_CKPT_DIR}",
                ),
            ))

        for cb in cbs:
            if hasattr(cb, "class_path") and "ModelCheckpoint" in cb.class_path:
                cb.init_args.dirpath = f"{root_dir}/{_CKPT_DIR}"


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
        **{sub: {"default_config_files": ["graphids/config/trainer.yaml"]}
           for sub in ("fit", "validate", "test", "predict")},
    },
)
