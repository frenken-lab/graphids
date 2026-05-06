"""Supervised MLP baseline: binary classification from flattened fusion features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from .base import FusionModuleBase, flatten_features


class MLPFusionModule(FusionModuleBase):
    """Same features as DQN, trained with BCE instead of RL."""

    automatic_optimization = True

    def __init__(
        self,
        state_dim: int = 18,
        hidden_dims: tuple[int, ...] = (64, 32),
        lr: float = 1e-3,
        decision_threshold: float = 0.5,
    ):
        super().__init__(state_dim=state_dim, decision_threshold=decision_threshold)
        self._store_init_kwargs(locals())

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward_scores(self, td: TensorDict) -> torch.Tensor:
        x = flatten_features(td).to(self.device)
        return torch.sigmoid(self.model(x).squeeze(-1))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
