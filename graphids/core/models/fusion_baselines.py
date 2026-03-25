"""Supervised fusion baselines: MLP and weighted average.

These consume the same 15-D state vector as the DQN agent but train with
standard supervised losses instead of RL episodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from ._training import binary_test_metrics


@dataclass(frozen=True)
class FusionResult:
    """Artifacts from fusion evaluation: predictions, scores, q-values."""
    preds: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    q_values: np.ndarray


def run_fusion_inference(agent, cache: dict) -> FusionResult:
    """Run fusion inference (works for both DQN and bandit agents)."""
    states = cache["states"]
    labels_t = cache["labels"]
    result = agent.predict(states)
    qv = agent.q_values(result["norm_states"])
    return FusionResult(
        preds=result["preds"].numpy(), labels=labels_t.numpy(),
        scores=result["fused_scores"].numpy(), q_values=qv.numpy(),
    )


class MLPFusionNetwork(nn.Module):
    """Simple MLP for binary classification from fusion state vectors."""

    def __init__(self, state_dim: int, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _SupervisedFusionOverrides:
    """Mixin providing trainer_overrides for supervised fusion modules (MLP, WeightedAvg)."""

    def trainer_overrides(self, cfg, dm) -> dict:
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        return {
            "default_root_dir": ".",
            "max_epochs": cfg.fusion.mlp_max_epochs,
            "callbacks": [
                ModelCheckpoint(
                    dirpath=".", filename="best_model",
                    monitor="val_loss", mode="min", save_top_k=1,
                ),
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ],
            "logger": pl.loggers.CSVLogger(save_dir=".", name="", version=""),
        }


class MLPFusionModule(_SupervisedFusionOverrides, pl.LightningModule):
    """Supervised MLP baseline: binary classification from fusion state vectors.

    Same state as DQN, but trained with BCE loss via Lightning instead of RL episodes.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        lr: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPFusionNetwork(state_dim, hidden_dims)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.test_metrics = binary_test_metrics()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, labels = batch
        logits = self(states)
        loss = self.loss_fn(logits, labels.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        logits = self(states)
        loss = self.loss_fn(logits, labels.float())
        preds = (logits > 0).long()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        states, labels = batch
        logits = self(states)
        preds = (logits > 0).long()
        self.test_metrics.update(preds, labels)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def fuse(self, state_features: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            logit = self(t)
            return 1 if logit.item() > 0 else 0


class WeightedAvgModule(_SupervisedFusionOverrides, pl.LightningModule):
    """Simplest baseline: learns a single scalar alpha per model.

    If this matches DQN's F1, the RL approach is unjustified.
    Fusion: score = (1 - sigmoid(w)) * vgae_conf + sigmoid(w) * gat_conf
    """

    def __init__(self, lr: float = 0.01, decision_threshold: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.weight = nn.Parameter(torch.zeros(1))
        self.loss_fn = nn.BCELoss()
        self.lr = lr
        self.decision_threshold = decision_threshold
        self.test_metrics = binary_test_metrics()

        from .registry import feature_layout

        layout = feature_layout()
        self._vgae_conf_idx = layout["vgae"].confidence_idx
        self._gat_conf_idx = layout["gat"].confidence_idx

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.weight)
        vgae_conf = states[:, self._vgae_conf_idx]
        gat_conf = states[:, self._gat_conf_idx]
        return torch.clamp((1 - alpha) * vgae_conf + alpha * gat_conf, 1e-7, 1 - 1e-7)

    def training_step(self, batch, batch_idx):
        states, labels = batch
        scores = self(states)
        loss = self.loss_fn(scores, labels.float())
        self.log("train_loss", loss, prog_bar=True)
        self.log("alpha", torch.sigmoid(self.weight).item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        scores = self(states)
        loss = self.loss_fn(scores, labels.float())
        preds = (scores > self.decision_threshold).long()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        states, labels = batch
        scores = self(states)
        preds = (scores > self.decision_threshold).long()
        self.test_metrics.update(preds, labels)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def fuse(self, state_features: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            alpha = torch.sigmoid(self.weight).item()
            vgae_conf = state_features[self._vgae_conf_idx]
            gat_conf = state_features[self._gat_conf_idx]
            score = (1 - alpha) * vgae_conf + alpha * gat_conf
            return 1 if score > self.decision_threshold else 0


class RLFusionModule(pl.LightningModule):
    """Lightning wrapper for RL fusion agents (DQN, bandit).

    Uses manual optimization. Both agents implement ``train_episode(states, labels)``
    returning a metrics dict. All returned keys are logged automatically.

    Constructor accepts config + method so Lightning can round-trip through
    ``save_hyperparameters`` / ``load_from_checkpoint``.
    """

    def __init__(self, cfg, method: str = "dqn", device: str = "cpu"):
        super().__init__()
        from omegaconf import OmegaConf

        # Normalize cfg for load_from_checkpoint round-trip (arrives as dict)
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        # Save hparams: cfg stored as plain dict for serialization
        self.save_hyperparameters(ignore=["cfg"])
        self.save_hyperparameters(
            {"cfg": OmegaConf.to_container(cfg) if hasattr(cfg, "_metadata") else cfg}
        )

        self.automatic_optimization = False
        self.cfg = cfg

        # Build agent from config
        if method == "dqn":
            from .dqn import EnhancedDQNFusionAgent

            agent = EnhancedDQNFusionAgent.from_config(cfg, device=device)
        elif method == "bandit":
            from .bandit import NeuralLinUCBAgent

            agent = NeuralLinUCBAgent.from_config(cfg, device=device)
        else:
            raise ValueError(f"RLFusionModule only handles dqn/bandit, got: {method}")

        self._optimizer_attr = "optimizer" if method == "dqn" else "backbone_optimizer"
        self.agent = agent
        self.test_metrics = binary_test_metrics()

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

    def on_save_checkpoint(self, checkpoint):
        checkpoint["agent_state"] = self.agent.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "agent_state" in checkpoint:
            self.agent.load_checkpoint(checkpoint["agent_state"])

    def trainer_overrides(self, cfg, dm) -> dict:
        """Trainer overrides for RL fusion training."""
        import math
        from pytorch_lightning.callbacks import ModelCheckpoint
        return {
            "default_root_dir": ".",
            "max_epochs": math.ceil(cfg.fusion.episodes / dm.steps_per_epoch),
            "callbacks": [ModelCheckpoint(
                dirpath=".", filename="best_model",
                monitor="val_acc", mode="max", save_top_k=1,
            )],
            "logger": pl.loggers.CSVLogger(save_dir=".", name="", version=""),
            "val_check_interval": min(50, dm.steps_per_epoch),
        }

    def configure_optimizers(self):
        return getattr(self.agent, self._optimizer_attr)


def build_fusion_module(cfg, device: torch.device) -> pl.LightningModule:
    """Build a fusion Lightning module for training (no checkpoint)."""
    method = cfg.fusion.method
    if method == "dqn":
        return RLFusionModule(cfg, method="dqn", device=str(device))
    elif method == "bandit":
        return RLFusionModule(cfg, method="bandit", device=str(device))
    elif method == "mlp":
        from .registry import fusion_state_dim
        return MLPFusionModule(state_dim=fusion_state_dim(), hidden_dims=cfg.fusion.mlp_hidden_dims, lr=cfg.fusion.lr)
    elif method == "weighted_avg":
        return WeightedAvgModule(lr=cfg.fusion.lr, decision_threshold=cfg.fusion.decision_threshold)
    else:
        raise ValueError(f"Unknown fusion method: {method}")


