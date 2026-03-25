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


class MLPFusionModule(pl.LightningModule):
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

    def save_checkpoint(self, path: str) -> None:
        torch.save({"model": self.model.state_dict()}, path)

    @classmethod
    def from_checkpoint(cls, ckpt: dict, cfg) -> MLPFusionModule:
        """Construct and load from a checkpoint dict."""
        from .registry import fusion_state_dim
        module = cls(state_dim=fusion_state_dim(), hidden_dims=cfg.fusion.mlp_hidden_dims, lr=cfg.fusion.lr)
        module.model.load_state_dict(ckpt["model"])
        return module

    def fuse(self, state_features: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            logit = self(t)
            return 1 if logit.item() > 0 else 0


class WeightedAvgModule(pl.LightningModule):
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

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.state_dict_for_save(), path)

    def state_dict_for_save(self) -> dict:
        return {"weight": self.weight.detach().cpu(), "alpha": torch.sigmoid(self.weight).item()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, cfg) -> WeightedAvgModule:
        """Construct and load from a checkpoint dict."""
        module = cls(lr=cfg.fusion.lr, decision_threshold=cfg.fusion.decision_threshold)
        module.weight.data = ckpt["weight"]
        return module

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
    """

    def __init__(self, agent, optimizer_attr: str = "optimizer"):
        super().__init__()
        self.automatic_optimization = False
        self.agent = agent
        self._optimizer_attr = optimizer_attr
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

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.agent.state_dict(), path)

    def configure_optimizers(self):
        return getattr(self.agent, self._optimizer_attr)

    @classmethod
    def from_checkpoint(cls, ckpt: dict, cfg, *, device: str = "cpu") -> RLFusionModule:
        """Construct RL agent from checkpoint and wrap in RLFusionModule."""
        method = cfg.fusion.method
        if method == "dqn":
            from .dqn import EnhancedDQNFusionAgent
            agent = EnhancedDQNFusionAgent.from_config(cfg, device=device, inference=True)
            agent.load_checkpoint(ckpt)
            return cls(agent, "optimizer")
        elif method == "bandit":
            from .bandit import NeuralLinUCBAgent
            agent = NeuralLinUCBAgent.from_config(cfg, device=device)
            agent.load_checkpoint(ckpt)
            return cls(agent, "backbone_optimizer")
        else:
            raise ValueError(f"RLFusionModule.from_checkpoint only handles dqn/bandit, got: {method}")


def build_fusion_module(cfg, device: torch.device) -> pl.LightningModule:
    """Build a fusion Lightning module for training (no checkpoint)."""
    method = cfg.fusion.method
    if method == "dqn":
        from .dqn import EnhancedDQNFusionAgent
        agent = EnhancedDQNFusionAgent.from_config(cfg, device=str(device))
        return RLFusionModule(agent, "optimizer")
    elif method == "bandit":
        from .bandit import NeuralLinUCBAgent
        agent = NeuralLinUCBAgent.from_config(cfg, device=str(device))
        return RLFusionModule(agent, "backbone_optimizer")
    elif method == "mlp":
        from .registry import fusion_state_dim
        return MLPFusionModule(state_dim=fusion_state_dim(), hidden_dims=cfg.fusion.mlp_hidden_dims, lr=cfg.fusion.lr)
    elif method == "weighted_avg":
        return WeightedAvgModule(lr=cfg.fusion.lr, decision_threshold=cfg.fusion.decision_threshold)
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def load_fusion_module(ckpt: dict, cfg, *, device: str = "cpu") -> pl.LightningModule:
    """Load a trained fusion module from checkpoint (any method)."""
    method = cfg.fusion.method
    if method in ("dqn", "bandit"):
        return RLFusionModule.from_checkpoint(ckpt, cfg, device=device)
    elif method == "mlp":
        return MLPFusionModule.from_checkpoint(ckpt, cfg)
    elif method == "weighted_avg":
        return WeightedAvgModule.from_checkpoint(ckpt, cfg)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
