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

    # Standard supervised training — trainer handles backward + step
    automatic_optimization = True

    def __init__(
        self,
        lr: float = 0.01,
        decision_threshold: float = 0.5,
    ):
        super().__init__(decision_threshold=decision_threshold)
        self._store_init_kwargs(locals())
        self.weight = nn.Parameter(torch.zeros(1))
        self.loss_fn = nn.BCELoss()

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
        self.log("train_loss", loss)
        self.log("alpha", torch.sigmoid(self.weight).item())
        return loss

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        scores = self(states)
        loss = self.loss_fn(scores, labels.float())
        preds = (scores > self.decision_threshold).long()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        states, labels = batch
        p1 = self(states)
        self.test_metrics.update(torch.stack([1.0 - p1, p1], dim=1), labels)

    def build_optimizers(self, max_epochs: int):
        return optim.Adam(self.parameters(), lr=self.lr), None
