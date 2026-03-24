"""Fusion stage: combines VGAE + GAT predictions via configurable method (DQN, MLP, weighted_avg)."""

from __future__ import annotations

import gc
import math
import structlog
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from graphids.core.preprocessing import CANBusDataModule

from .trainer_factory import build_module, load_model, make_trainer

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


# ---------------------------------------------------------------------------
# DQN Lightning module (train + eval)
# ---------------------------------------------------------------------------


class RLFusionModule(pl.LightningModule):
    """Lightning wrapper for RL fusion agents (DQN, bandit).

    Uses manual optimization. Both agents implement ``train_episode(states, labels)``
    returning a metrics dict. All returned keys are logged automatically.
    """

    def __init__(self, agent, optimizer_attr: str = "optimizer"):
        super().__init__()
        self.automatic_optimization = False
        self.agent = agent
        self._optimizer_attr = optimizer_attr
        from graphids.core.models.registry import fusion_test_metrics
        self.test_metrics = fusion_test_metrics()

    def training_step(self, batch, batch_idx):
        states, labels = batch
        result = self.agent.train_episode(states, labels)
        for k, v in result.items():
            if v is not None:
                self.log(k, float(v), prog_bar=(k in ("avg_reward", "accuracy")))

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        metrics = self.agent.validate_batch(states, labels)
        self.log("val_acc", metrics.get("accuracy", 0.0), prog_bar=True)

    def test_step(self, batch, batch_idx):
        states, labels = batch
        result = self.agent.predict(states)
        self.test_metrics.update(result["preds"], labels)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        return getattr(self.agent, self._optimizer_attr)


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


# ---------------------------------------------------------------------------
# Per-method training functions
# ---------------------------------------------------------------------------


def train_fusion(cfg) -> dict:
    """Train fusion agent on cached VGAE+GAT predictions. Returns result dict with checkpoint and metrics."""
    pl.seed_everything(cfg.seed)
    dm = FusionDataModule(cfg)
    dm.setup("fit")
    device = dm.device

    # Build module via factory
    method = cfg.fusion.method
    module = build_module(cfg, "fusion", device)
    is_rl = isinstance(module, RLFusionModule)

    # Save function per module type (custom format read by eval stage)
    if is_rl:
        save_fn = lambda: torch.save(module.agent.state_dict(), "best_model.pt")
    elif method == "mlp":
        save_fn = lambda: torch.save({"model": module.model.state_dict()}, "best_model.pt")
    else:  # weighted_avg
        save_fn = lambda: torch.save(module.state_dict_for_save(), "best_model.pt")

    # Build trainer: RL monitors val_acc, baselines monitor val_loss with early stopping
    if is_rl:
        trainer = make_trainer(cfg, "fusion",
            default_root_dir=".",
            max_epochs=math.ceil(cfg.fusion.episodes / dm.steps_per_epoch),
            callbacks=[ModelCheckpoint(dirpath=".", filename="best_model", monitor="val_acc", mode="max", save_top_k=1)],
            logger=pl.loggers.CSVLogger(save_dir=".", name="", version=""),
            val_check_interval=min(50, dm.steps_per_epoch),
        )
    else:
        from pytorch_lightning.callbacks import EarlyStopping
        trainer = make_trainer(cfg, "fusion",
            default_root_dir=".",
            max_epochs=cfg.fusion.mlp_max_epochs,
            callbacks=[
                ModelCheckpoint(dirpath=".", filename="best_model", monitor="val_loss", mode="min", save_top_k=1),
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ],
            logger=pl.loggers.CSVLogger(save_dir=".", name="", version=""),
        )

    trainer.fit(module, datamodule=dm)
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        module.load_state_dict(torch.load(best_path, weights_only=True)["state_dict"])
    save_fn()
    best_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()

    metrics = {"best_acc": best_acc, "fusion_method": method}
    log.info("saved_fusion", method=method, checkpoint="best_model.pt", best_acc=round(best_acc, 4))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"checkpoint": "best_model.pt", "metrics": metrics}
