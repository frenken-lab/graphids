"""
CLI entry point: LightningCLI for training, jsonargparse for manifest ops.
"""

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

    LightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        seed_everything_default=42,
        parser_kwargs={"default_env": True, "env_prefix": "KD_GAT"},
    )
