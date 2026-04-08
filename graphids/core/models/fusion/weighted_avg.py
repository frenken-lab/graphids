"""Simplest fusion baseline: learns a single scalar alpha per model.

If this matches DQN's F1, the RL approach is unjustified.
Fusion: score = (1 - sigmoid(w)) * vgae_conf + sigmoid(w) * gat_conf
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .base import LAYOUT, FusionModuleBase


class WeightedAvgModule(FusionModuleBase):
    """Learns a single scalar alpha blending VGAE and GAT confidence."""

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

        self._vgae_conf_idx = LAYOUT["vgae"].confidence_idx
        self._gat_conf_idx = LAYOUT["gat"].confidence_idx

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
