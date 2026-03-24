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

from .trainer_factory import load_model, make_trainer

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

    Uses manual optimization. The agent-specific training logic is passed
    as a callable ``train_fn(module, batch)`` to keep the module generic.
    Validation and test are identical for all RL agents.
    """

    def __init__(self, agent, train_fn, optimizer_attr: str = "optimizer", cfg=None):
        super().__init__()
        self.automatic_optimization = False
        self.agent = agent
        self._train_fn = train_fn
        self._optimizer_attr = optimizer_attr
        self.cfg = cfg
        from graphids.core.models.registry import fusion_test_metrics
        self.test_metrics = fusion_test_metrics()

    def training_step(self, batch, batch_idx):
        self._train_fn(self, batch)

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


def _dqn_train_step(module: RLFusionModule, batch) -> None:
    """DQN episode: delegates to agent.train_episode(), logs metrics."""
    states, labels = batch
    result = module.agent.train_episode(states, labels)
    module.log("train_reward", result["avg_reward"], prog_bar=True)
    module.log("epsilon", result["epsilon"])
    if result["loss"] is not None:
        module.log("train_loss", result["loss"])


def _bandit_train_step(module: RLFusionModule, batch) -> None:
    """Bandit episode: train_episode + regret logging."""
    states, labels = batch
    result = module.agent.train_episode(states, labels)
    module.log("train_acc", result["accuracy"], prog_bar=True)
    module.log("avg_ucb_width", module.agent.regret_stats()["avg_ucb_width"])


def _make_fusion_dataloaders(train_cache, val_cache, batch_size: int):
    """Build train/val DataLoaders from cached prediction tensors."""
    train_ds = TensorDataset(train_cache["states"], train_cache["labels"])
    val_ds = TensorDataset(val_cache["states"], val_cache["labels"])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


# ---------------------------------------------------------------------------
# Per-method training functions
# ---------------------------------------------------------------------------


def train_fusion(cfg) -> dict:
    """Train fusion agent on cached VGAE+GAT predictions. Returns result dict with checkpoint and metrics."""
    pl.seed_everything(cfg.seed)
    dm = CANBusDataModule.from_cfg(cfg)
    dm.setup("fit")
    dm.populate_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load frozen VGAE + GAT, cache predictions, release models
    vgae = load_model(cfg, "vgae", "autoencoder", device)
    gat = load_model(cfg, "gat", cfg.gat_stage, device)
    models = {"vgae": vgae, "gat": gat}
    train_cache = cache_predictions(models, list(dm.train_dataset), device, cfg.fusion.max_samples, batch_size=cfg.evaluation.batch_size)
    val_cache = cache_predictions(models, list(dm.val_dataset), device, cfg.fusion.max_val_samples, batch_size=cfg.evaluation.batch_size)
    del vgae, gat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build module + save function per method
    method = cfg.fusion.method
    if method == "dqn":
        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        agent = EnhancedDQNFusionAgent.from_config(cfg, device=str(device))
        module = RLFusionModule(agent, _dqn_train_step, "optimizer", cfg)
        save_fn = lambda: torch.save(agent.state_dict(), "best_model.pt")
        is_rl = True
    elif method == "bandit":
        from graphids.core.models.bandit import NeuralLinUCBAgent
        agent = NeuralLinUCBAgent.from_config(cfg, device=str(device))
        module = RLFusionModule(agent, _bandit_train_step, "backbone_optimizer", cfg)
        save_fn = lambda: torch.save(agent.state_dict(), "best_model.pt")
        is_rl = True
    elif method == "mlp":
        from graphids.core.models.fusion_baselines import MLPFusionModule
        from graphids.core.models.registry import fusion_state_dim
        module = MLPFusionModule(state_dim=fusion_state_dim(), hidden_dims=cfg.fusion.mlp_hidden_dims, lr=cfg.fusion.lr)
        save_fn = lambda: torch.save({"model": module.model.state_dict()}, "best_model.pt")
        is_rl = False
    elif method == "weighted_avg":
        from graphids.core.models.fusion_baselines import WeightedAvgModule
        module = WeightedAvgModule(lr=cfg.fusion.lr, decision_threshold=cfg.fusion.decision_threshold)
        save_fn = lambda: torch.save(module.state_dict_for_save(), "best_model.pt")
        is_rl = False
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    # Shared training: RL methods use episode sampling, baselines use batch sampling
    if is_rl:
        train_dl, val_dl = _make_fusion_dataloaders(
            train_cache, {k: v[:cfg.fusion.max_val_samples] for k, v in val_cache.items()},
            cfg.fusion.episode_sample_size,
        )
        steps_per_epoch = math.ceil(len(train_cache["states"]) / cfg.fusion.episode_sample_size)
        trainer = make_trainer(cfg, "fusion",
            default_root_dir=".",
            max_epochs=math.ceil(cfg.fusion.episodes / steps_per_epoch),
            callbacks=[ModelCheckpoint(dirpath=".", filename="best_model", monitor="val_acc", mode="max", save_top_k=1)],
            logger=pl.loggers.CSVLogger(save_dir=".", name="", version=""),
            val_check_interval=min(50, steps_per_epoch),
        )
    else:
        from pytorch_lightning.callbacks import EarlyStopping
        train_dl, val_dl = _make_fusion_dataloaders(train_cache, val_cache, cfg.dqn.batch_size)
        trainer = make_trainer(cfg, "fusion",
            default_root_dir=".",
            max_epochs=cfg.fusion.mlp_max_epochs,
            callbacks=[
                ModelCheckpoint(dirpath=".", filename="best_model", monitor="val_loss", mode="min", save_top_k=1),
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ],
            logger=pl.loggers.CSVLogger(save_dir=".", name="", version=""),
        )

    trainer.fit(module, train_dl, val_dl)
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
