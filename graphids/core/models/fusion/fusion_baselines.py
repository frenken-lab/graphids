"""Fusion Lightning module base + supervised baselines (MLP, weighted average).

``FusionModuleBase`` — shared ``test_metrics`` bookkeeping plus delegating
``test_step`` / ``validation_step`` used by the RL subclasses
(``BanditFusionModule``, ``DQNFusionModule``). Those subclasses implement
``predict`` / ``validate_batch`` and let the base handle the rest.

``MLPFusionModule`` and ``WeightedAvgModule`` are supervised baselines: they
override ``training_step`` / ``validation_step`` / ``test_step`` with standard
loss-based flows and inherit only the metric hooks from the base.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from .._training import binary_test_metrics
from .fusion_features import fusion_state_dim

# Module-level constant so ``--print_config`` serializes the real default
# instead of a sentinel (see memory note: no --print_config null serialization).
_DEFAULT_STATE_DIM = fusion_state_dim()


class FusionModuleBase(pl.LightningModule):
    """Base Lightning wrapper for fusion models with shared metric bookkeeping.

    RL subclasses (Bandit, DQN) rely on the default ``test_step`` /
    ``validation_step`` which delegate to ``predict`` / ``validate_batch``.
    Supervised subclasses (MLP, WeightedAvg) override both with standard
    loss-based flows but still inherit the epoch hooks.
    """

    def __init__(self):
        super().__init__()
        self.test_metrics = binary_test_metrics()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        states, labels = batch
        result = self.predict(states)
        self.test_metrics.update(result["preds"], labels)

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        metrics = self.validate_batch(states, labels)
        self.log("val_acc", metrics.get("accuracy", 0.0), prog_bar=True)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())


class MLPFusionModule(FusionModuleBase):
    """Supervised MLP baseline: binary classification from fusion state vectors.

    Same state as DQN, but trained with BCE loss via Lightning instead of RL episodes.
    """

    def __init__(
        self,
        state_dim: int = _DEFAULT_STATE_DIM,
        hidden_dims: tuple[int, ...] = (64, 32),
        lr: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        states, labels = batch
        logits = self(states)
        preds = (logits > 0).long()
        self.test_metrics.update(preds, labels)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class WeightedAvgModule(FusionModuleBase):
    """Simplest baseline: learns a single scalar alpha per model.

    If this matches DQN's F1, the RL approach is unjustified.
    Fusion: score = (1 - sigmoid(w)) * vgae_conf + sigmoid(w) * gat_conf
    """

    def __init__(
        self,
        lr: float = 0.01,
        decision_threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.weight = nn.Parameter(torch.zeros(1))
        self.loss_fn = nn.BCELoss()
        self.lr = lr
        self.decision_threshold = decision_threshold

        from .fusion_features import feature_layout
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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        states, labels = batch
        scores = self(states)
        preds = (scores > self.decision_threshold).long()
        self.test_metrics.update(preds, labels)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
