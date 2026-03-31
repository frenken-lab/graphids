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
        # Allowlist: adding a new optimizer/scheduler requires editing here, not just YAML
        parser.add_optimizer_args((torch.optim.Adam, torch.optim.AdamW))
        parser.add_lr_scheduler_args((torch.optim.lr_scheduler.CosineAnnealingLR,
                                      torch.optim.lr_scheduler.ReduceLROnPlateau))
        parser.link_arguments("data.init_args.dataset", "model.init_args.dataset")
        parser.link_arguments("data.init_args.lake_root", "model.init_args.lake_root")
        parser.link_arguments("seed_everything", "model.init_args.seed")
        parser.link_arguments("model.init_args.conv_type", "data.init_args.conv_type")
        parser.link_arguments("model.init_args.heads", "data.init_args.heads")

    def before_instantiate_classes(self):
        """Patch parsed config before classes are constructed.

        Fixes two issues that can't be expressed via link_arguments:
        1. CSVLogger save_dir — link can't target a list element
        2. CurriculumEpochCallback — only needed for CurriculumDataModule
        """
        if not self.subcommand:
            return  # no-op when CLI created without subcommand (e.g. validate)
        subcfg = self.config[self.subcommand]

        # 1. Propagate default_root_dir → CSVLogger save_dir
        root_dir = subcfg.trainer.default_root_dir
        loggers = subcfg.trainer.logger
        if root_dir and isinstance(loggers, list):
            for lg in loggers:
                if hasattr(lg, "class_path") and "CSVLogger" in lg.class_path:
                    lg.init_args.save_dir = root_dir

        # 2. Drop CurriculumEpochCallback for non-curriculum DataModules
        data_cp = getattr(subcfg.data, "class_path", "")
        if "CurriculumDataModule" not in data_cp:
            cbs = subcfg.trainer.callbacks
            if isinstance(cbs, list):
                subcfg.trainer.callbacks = [
                    cb for cb in cbs
                    if not (hasattr(cb, "class_path")
                            and "CurriculumEpochCallback" in cb.class_path)
                ]


CLI_KWARGS = dict(
    model_class=pl.LightningModule,
    datamodule_class=pl.LightningDataModule,
    subclass_mode_model=True,
    subclass_mode_data=True,
    # Optimizer wiring: YAML `optimizer:` block → CLI auto-configures.
    # No `optimizer:` block → model's own configure_optimizers() wins.
    # Fusion models (automatic_optimization=False) manage their own optimizers.
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
