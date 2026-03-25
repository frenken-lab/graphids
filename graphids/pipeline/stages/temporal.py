"""Temporal data module: wraps CANBusDataModule with temporal sequence grouping.

The TemporalDataModule is the only export — training runs via the generic train_stage.
"""

from __future__ import annotations

import structlog

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from graphids.core.preprocessing import CANBusDataModule
from graphids.core.preprocessing._temporal import (
    TemporalGraphDataset,
    TemporalGrouper,
    collate_temporal,
)

from .trainer_factory import load_model

log = structlog.get_logger()


class TemporalDataModule(pl.LightningDataModule):
    """Loads graph data, groups into temporal sequences, serves DataLoaders.

    During setup, loads a pretrained GAT to probe the spatial embedding dim.
    The GAT reference is stored for build_module to consume (avoids loading twice).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.spatial_dim: int | None = None
        self.gat: torch.nn.Module | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    def setup(self, stage=None):
        raw_dm = CANBusDataModule.from_cfg(self.cfg)
        raw_dm.setup("fit")
        raw_dm.populate_config(self.cfg)

        tc = self.cfg.temporal

        # Load pretrained GAT and probe spatial embedding dim
        self.gat = load_model(self.cfg, "gat", self.cfg.gat_stage, self._device)
        with torch.no_grad():
            probe = raw_dm.train_dataset[0].clone().to(self._device, non_blocking=True)
            _, emb = self.gat(probe, return_embedding=True)
            self.spatial_dim = emb.shape[-1]
        log.info("spatial_embedding_dim", dim=self.spatial_dim)

        # Contiguous time split: first train_split% train, rest val
        grouper = TemporalGrouper(window=tc.temporal_window, stride=tc.temporal_stride)
        all_graphs = list(raw_dm.train_dataset) + list(raw_dm.val_dataset)
        split_idx = int(len(all_graphs) * tc.train_split)

        self._train_sequences = grouper.group(all_graphs[:split_idx])
        self._val_sequences = grouper.group(all_graphs[split_idx:])

        log.info(
            "temporal_sequences",
            train=len(self._train_sequences),
            val=len(self._val_sequences),
            total_graphs=len(all_graphs),
        )

        if not self._train_sequences or not self._val_sequences:
            raise ValueError("Not enough graphs for temporal windowing")

        self._batch_size = (
            tc.batch_size if tc.batch_size > 0
            else max(1, min(32, len(self._train_sequences) // 10))
        )

    def train_dataloader(self):
        return DataLoader(
            TemporalGraphDataset(self._train_sequences, self._device),
            batch_size=self._batch_size, shuffle=True,
            collate_fn=collate_temporal, num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            TemporalGraphDataset(self._val_sequences, self._device),
            batch_size=self._batch_size, shuffle=False,
            collate_fn=collate_temporal, num_workers=0,
        )
