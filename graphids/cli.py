"""Shared LightningCLI subclass — single definition used by __main__ and expand."""
from __future__ import annotations

import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.loggers import WandbLogger


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
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.CosineAnnealingLR)
        parser.link_arguments("data.init_args.dataset", "model.init_args.dataset")
        parser.link_arguments("data.init_args.lake_root", "model.init_args.lake_root")
        parser.link_arguments("seed_everything", "model.init_args.seed")
        parser.link_arguments("model.init_args.conv_type", "data.init_args.conv_type")
        parser.link_arguments("model.init_args.heads", "data.init_args.heads")
        # CSVLogger save_dir must follow default_root_dir so metrics.csv lands in run dir
        parser.link_arguments("trainer.default_root_dir", "trainer.logger.init_args.save_dir")


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
