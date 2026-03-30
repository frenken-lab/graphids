"""Shared LightningCLI subclass — single definition used by __main__ and expand."""
from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI


class GraphIDSCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        #TODO: init_args are 4 of the 5 here, seed_everything is also init_args
        # should be organized between model, data, and other init_arg categories
        parser.link_arguments("data.init_args.dataset", "model.init_args.dataset")
        parser.link_arguments("data.init_args.lake_root", "model.init_args.lake_root")
        parser.link_arguments("seed_everything", "model.init_args.seed")
        parser.link_arguments("model.init_args.conv_type", "data.init_args.conv_type")
        parser.link_arguments("model.init_args.heads", "data.init_args.heads")


CLI_KWARGS = dict(
    model_class=pl.LightningModule,
    datamodule_class=pl.LightningDataModule,
    subclass_mode_model=True,
    subclass_mode_data=True,
    seed_everything_default=42,
    save_config_kwargs={"overwrite": True},
    parser_kwargs={
        "default_env": True,
        "env_prefix": "KD_GAT",
        **{sub: {"default_config_files": ["graphids/config/trainer.yaml"]}
           for sub in ("fit", "validate", "test", "predict")},
    },
)
