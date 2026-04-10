"""Supervised MLP baseline: binary classification from fusion state vectors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .base import STATE_DIM, FusionModuleBase


class MLPFusionModule(FusionModuleBase):
    """Same state as DQN, but trained with BCE loss instead of RL episodes."""

    # Standard supervised training — trainer handles backward + step
    automatic_optimization = True

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dims: tuple[int, ...] = (64, 32),
        lr: float = 0.001,
    ):
        super().__init__()
        self.hparams = self._capture_hparams(locals())

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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        logits = self(states)
        loss = self.loss_fn(logits, labels.float())
        preds = (logits > 0).long()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        states, labels = batch
        logits = self(states)
        preds = (logits > 0).long()
        self.test_metrics.update(preds, labels)

    def build_optimizers(self, max_epochs: int):
        return optim.Adam(self.parameters(), lr=self.lr), None
