"""Fusion stage: combines VGAE + GAT predictions via configurable method (DQN, MLP, weighted_avg)."""

from __future__ import annotations

import math
import structlog
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from graphids.core.preprocessing import CANBusDataModule

from .data_loading import cache_predictions, cleanup
from .trainer_factory import load_model

log = structlog.get_logger()


def _fusion_test_metrics() -> MetricCollection:
    return MetricCollection({
        "accuracy": BinaryAccuracy(), "f1": BinaryF1Score(),
        "precision": BinaryPrecision(), "recall": BinaryRecall(),
        "specificity": BinarySpecificity(), "auc": BinaryAUROC(),
    })


# ---------------------------------------------------------------------------
# DQN Lightning module (train + eval)
# ---------------------------------------------------------------------------


class DQNFusionModule(pl.LightningModule):
    """Lightning wrapper for DQN fusion agent.

    Uses manual optimization since the agent manages its own optimizer.
    training_step runs one RL episode: select actions, compute rewards,
    store experiences, gradient steps from replay buffer, epsilon decay.
    """

    def __init__(self, agent, cfg=None):
        super().__init__()
        self.automatic_optimization = False
        self.agent = agent
        self.cfg = cfg
        self.test_metrics = _fusion_test_metrics()

    def training_step(self, batch, batch_idx):
        states, labels = batch
        actions, alphas, norm_states = self.agent.select_action_batch(states, training=True)
        # TODO(open-question): Training uses (alpha > 0.5) as prediction, but
        # validation uses the proper fused score. See dqn.py top-level comment.
        preds = (alphas > 0.5).long()
        rewards = self.agent.reward_calc.compute(preds, labels, norm_states, alphas)
        self.agent.store_experiences_batch(norm_states, actions, rewards)

        # Gradient steps from replay buffer
        loss = None
        if self.agent.buffer_size_current >= self.cfg.dqn.batch_size:
            for _ in range(self.cfg.fusion.gpu_training_steps):
                loss = self.agent.train_step()

        # Epsilon decay
        self.agent.epsilon = max(self.agent.min_epsilon, self.agent.epsilon * self.agent.epsilon_decay)

        self.log("train_reward", rewards.mean(), prog_bar=True)
        self.log("epsilon", self.agent.epsilon)
        if loss is not None:
            self.log("train_loss", loss)

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        metrics = self.agent.validate_batch(states, labels)
        self.log("val_acc", metrics.get("accuracy", 0.0), prog_bar=True)

    def test_step(self, batch, batch_idx):
        states, labels = batch
        actions, alphas, norm_states = self.agent.select_action_batch(states, training=False)
        anomaly_scores, gat_probs = self.agent.reward_calc.derive_scores(norm_states)
        fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
        preds = (fused_scores > 0.5).long()
        self.test_metrics.update(preds, labels)
        self.log_dict(self.test_metrics, batch_size=len(labels))

    def configure_optimizers(self):
        return self.agent.optimizer


# ---------------------------------------------------------------------------
# Bandit Lightning module (train + eval)
# ---------------------------------------------------------------------------


class BanditFusionModule(pl.LightningModule):
    """Lightning wrapper for Neural-LinUCB bandit agent.

    Uses manual optimization — Sherman-Morrison updates are closed-form,
    with periodic backbone retraining via the agent's internal optimizer.
    """

    def __init__(self, agent, cfg=None):
        super().__init__()
        self.automatic_optimization = False
        self.agent = agent
        self.cfg = cfg
        self.test_metrics = _fusion_test_metrics()

    def training_step(self, batch, batch_idx):
        states, labels = batch
        result = self.agent.train_episode(states, labels)
        self.log("train_acc", result["accuracy"], prog_bar=True)
        regret = self.agent.regret_stats()
        self.log("avg_ucb_width", regret["avg_ucb_width"])

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        metrics = self.agent.validate_batch(states, labels)
        self.log("val_acc", metrics.get("accuracy", 0.0), prog_bar=True)

    def test_step(self, batch, batch_idx):
        states, labels = batch
        actions, alphas, norm_states = self.agent.select_action_batch(states, training=False)
        anomaly_scores, gat_probs = self.agent.reward_calc.derive_scores(norm_states)
        fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
        preds = (fused_scores > 0.5).long()
        self.test_metrics.update(preds, labels)
        self.log_dict(self.test_metrics, batch_size=len(labels))

    def configure_optimizers(self):
        return self.agent.backbone_optimizer


# ---------------------------------------------------------------------------
# Trainer factories
# ---------------------------------------------------------------------------


def _make_rl_fusion_trainer(cfg, steps_per_epoch: int):
    """Create Lightning Trainer for RL fusion (DQN/bandit).

    Maps episodes to epochs. Validates every 50 training steps.
    ModelCheckpoint saves best model by val_acc.
    """
    max_epochs = math.ceil(cfg.fusion.episodes / steps_per_epoch)
    return pl.Trainer(
        default_root_dir=".",
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[
            ModelCheckpoint(
                dirpath=".", filename="best_model",
                monitor="val_acc", mode="max", save_top_k=1,
            ),
        ],
        val_check_interval=min(50, steps_per_epoch),
        logger=pl.loggers.CSVLogger(save_dir=".", name="", version=""),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )


def _make_fusion_trainer(cfg):
    """Create a lightweight Lightning Trainer for fusion baselines (MLP/WeightedAvg)."""
    from pytorch_lightning.callbacks import EarlyStopping

    return pl.Trainer(
        default_root_dir=".",
        max_epochs=cfg.fusion.mlp_max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[
            ModelCheckpoint(
                dirpath=".", filename="best_model",
                monitor="val_loss", mode="min", save_top_k=1,
            ),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
        logger=pl.loggers.CSVLogger(save_dir=".", name="", version=""),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )


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


def _save_dqn_ckpt(agent) -> None:
    torch.save({
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "epsilon": agent.epsilon,
    }, "best_model.pt")


def _train_dqn_fusion(cfg, train_cache, val_cache, device) -> float:
    """DQN RL fusion via Lightning Trainer. Returns best validation accuracy."""
    from graphids.core.models.dqn import EnhancedDQNFusionAgent

    agent = EnhancedDQNFusionAgent.from_config(cfg, device=str(device))
    module = DQNFusionModule(agent, cfg)

    train_dl, val_dl = _make_fusion_dataloaders(
        train_cache,
        {k: v[:5000] for k, v in val_cache.items()},
        cfg.fusion.episode_sample_size,
    )
    steps_per_epoch = math.ceil(len(train_cache["states"]) / cfg.fusion.episode_sample_size)
    trainer = _make_rl_fusion_trainer(cfg, steps_per_epoch)
    trainer.fit(module, train_dl, val_dl)

    # Save in agent's native format (Lightning checkpoint is .ckpt)
    _save_dqn_ckpt(agent)
    return trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()


def _train_bandit_fusion(cfg, train_cache, val_cache, device) -> float:
    """Neural-LinUCB bandit fusion via Lightning Trainer. Returns best validation accuracy."""
    from graphids.core.models.bandit import NeuralLinUCBAgent

    agent = NeuralLinUCBAgent.from_config(cfg, device=str(device))
    module = BanditFusionModule(agent, cfg)

    train_dl, val_dl = _make_fusion_dataloaders(
        train_cache,
        {k: v[:5000] for k, v in val_cache.items()},
        cfg.fusion.episode_sample_size,
    )
    steps_per_epoch = math.ceil(len(train_cache["states"]) / cfg.fusion.episode_sample_size)
    trainer = _make_rl_fusion_trainer(cfg, steps_per_epoch)
    trainer.fit(module, train_dl, val_dl)

    # Save in agent's native format
    torch.save(agent.state_dict(), "best_model.pt")
    return trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()


def _train_mlp_fusion(cfg, train_cache, val_cache, device) -> float:
    """MLP supervised fusion via Lightning Trainer. Returns best validation accuracy."""
    from graphids.core.models.fusion_baselines import MLPFusionModule
    from graphids.core.models.registry import fusion_state_dim

    module = MLPFusionModule(
        state_dim=fusion_state_dim(),
        hidden_dims=cfg.fusion.mlp_hidden_dims,
        lr=cfg.fusion.lr,
    )
    train_dl, val_dl = _make_fusion_dataloaders(
        train_cache, val_cache, cfg.dqn.batch_size,
    )
    trainer = _make_fusion_trainer(cfg)
    trainer.fit(module, train_dl, val_dl)

    torch.save({"model": module.model.state_dict()}, "best_model.pt")
    best_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()
    return best_acc


def _train_weighted_avg_fusion(cfg, train_cache, val_cache, device) -> float:
    """Weighted average fusion via Lightning Trainer. Returns best validation accuracy."""
    from graphids.core.models.fusion_baselines import WeightedAvgModule

    module = WeightedAvgModule(lr=cfg.fusion.lr)
    train_dl, val_dl = _make_fusion_dataloaders(
        train_cache, val_cache, cfg.dqn.batch_size,
    )
    trainer = _make_fusion_trainer(cfg)
    trainer.fit(module, train_dl, val_dl)

    torch.save(module.state_dict_for_save(), "best_model.pt")
    best_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()
    return best_acc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_fusion(cfg) -> dict:
    """Train fusion agent on cached VGAE+GAT predictions. Returns result dict with checkpoint and metrics."""
    pl.seed_everything(cfg.seed)
    dm = CANBusDataModule.from_cfg(cfg)
    dm.setup("fit")
    dm.populate_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load frozen VGAE + GAT
    vgae = load_model(cfg, "vgae", "autoencoder", device)
    gat = load_model(cfg, "gat", "curriculum", device)

    # Cache predictions
    log.info("Caching VGAE + GAT predictions ...")
    models = {"vgae": vgae, "gat": gat}
    train_cache = cache_predictions(models, list(dm.train_dataset), device, cfg.fusion.max_samples)
    val_cache = cache_predictions(models, list(dm.val_dataset), device, cfg.fusion.max_val_samples)
    del vgae, gat
    cleanup()

    # Dispatch on fusion method
    method = cfg.fusion.method
    if method == "dqn":
        best_acc = _train_dqn_fusion(cfg, train_cache, val_cache, device)
    elif method == "bandit":
        best_acc = _train_bandit_fusion(cfg, train_cache, val_cache, device)
    elif method == "mlp":
        best_acc = _train_mlp_fusion(cfg, train_cache, val_cache, device)
    elif method == "weighted_avg":
        best_acc = _train_weighted_avg_fusion(cfg, train_cache, val_cache, device)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    ckpt = Path("best_model.pt")
    # config.yaml already saved by run_stage() in __init__.py

    metrics = {"best_acc": best_acc, "val_loss": 1.0 - best_acc, "fusion_method": method}
    log.info("saved_fusion", method=method, checkpoint=str(ckpt), best_acc=round(best_acc, 4))
    cleanup()
    return {"checkpoint": str(ckpt), "metrics": metrics}
