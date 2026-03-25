"""Fusion stage: combines VGAE + GAT predictions via configurable method (DQN, MLP, weighted_avg)."""

from __future__ import annotations

import gc
import math
import structlog

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

import pytorch_lightning as pl
from graphids.core.preprocessing import CANBusDataModule

from .trainer_factory import load_model

log = structlog.get_logger()


def cache_predictions(models: dict[str, nn.Module], data, device, max_samples: int = 150_000, batch_size: int = 256):
    """Run registered extractors over data, produce N-D state vectors for DQN.

    Uses a DataLoader for batched clone+transfer, then extracts per-graph
    features within each on-device batch (extractors are not batch-aware).
    """
    from graphids.core.models.registry import extractors as registry_extractors
    from graphids.core.preprocessing import get_batch_index

    active = [(name, ext) for name, ext in registry_extractors() if name in models]
    for model in models.values():
        model.eval()

    capped = data[:max_samples]
    loader = PyGDataLoader(capped, batch_size=batch_size, shuffle=False)

    states, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            for g in batch.to_data_list():
                batch_idx = get_batch_index(g, device)
                features = [ext.extract(models[name], g, batch_idx, device) for name, ext in active]
                states.append(torch.cat(features))
                labels.append(g.y[0] if g.y.dim() > 0 else g.y)

    return {"states": torch.stack(states), "labels": torch.tensor(labels)}


class FusionDataModule(pl.LightningDataModule):
    """Loads frozen VGAE+GAT, caches state vectors, serves DataLoaders.

    Wraps CANBusDataModule internally — callers never touch raw graph data.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        is_rl = cfg.fusion.method in ("dqn", "bandit")
        self._batch_size = cfg.fusion.episode_sample_size if is_rl else cfg.dqn.batch_size
        self.train_cache: dict | None = None
        self.val_cache: dict | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(len(self.train_cache["states"]) / self._batch_size)

    def setup(self, stage=None):
        raw_dm = CANBusDataModule.from_cfg(self.cfg)
        raw_dm.setup("fit")
        raw_dm.populate_config(self.cfg)

        vgae = load_model(self.cfg, "vgae", "autoencoder", self._device)
        gat = load_model(self.cfg, "gat", self.cfg.gat_stage, self._device)
        models = {"vgae": vgae, "gat": gat}
        bs = self.cfg.evaluation.batch_size
        self.train_cache = cache_predictions(models, list(raw_dm.train_dataset), self._device, self.cfg.fusion.max_samples, batch_size=bs)
        self.val_cache = cache_predictions(models, list(raw_dm.val_dataset), self._device, self.cfg.fusion.max_val_samples, batch_size=bs)

        del vgae, gat, models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_dataloader(self):
        ds = TensorDataset(self.train_cache["states"], self.train_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        ds = TensorDataset(self.val_cache["states"], self.val_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size)
