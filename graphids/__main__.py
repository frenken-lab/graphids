"""CLI entry point: LightningCLI with linked args for DRY config."""

from __future__ import annotations

import torch
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
mp.set_sharing_strategy("file_system")

import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    cache_logger_on_first_use=True,
)

if __name__ == "__main__":
    import pytorch_lightning as pl
    from pytorch_lightning.cli import LightningCLI

    class GraphIDSCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            # Set once on data, auto-copied to model — no double-passing.
            parser.link_arguments("data.init_args.dataset", "model.init_args.dataset")
            parser.link_arguments("data.init_args.lake_root", "model.init_args.lake_root")
            parser.link_arguments("seed_everything", "model.init_args.seed")
            # NOT linking trainer.max_epochs → model.init_args.training.max_epochs.
            # That link mutates the training Namespace after jsonargparse instantiation,
            # converting the typed TrainingConfig dataclass back to a raw dict/Namespace.
            # max_epochs lives in trainer.max_epochs (controls loop) and separately in
            # model.training.max_epochs (controls CosineAnnealingLR T_max).

    GraphIDSCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        seed_everything_default=42,
        parser_kwargs={"default_env": True, "env_prefix": "KD_GAT"},
    )
