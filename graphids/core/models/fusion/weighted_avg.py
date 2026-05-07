"""Simplest fusion baseline: learns a single scalar alpha blending vgae anomaly + gat attack prob.

score = (1-alpha) * vgae_anom + alpha * gat_attack
where vgae_anom = 1 - vgae_conf = recon_mean/(1+recon_mean)  (high = anomalous)
      gat_attack = gat/probs[:,1]                              (high = attack)

If this matches DQN's F1, the RL approach is unjustified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from .base import FusionModuleBase


class WeightedAvgModule(FusionModuleBase):
    """alpha = sigmoid(w); score = (1-alpha)·vgae_conf + alpha·gat_conf."""

    automatic_optimization = True

    def __init__(self, lr: float = 1e-2, decision_threshold: float = 0.5, state_dim: int = 18):
        super().__init__(state_dim=state_dim, decision_threshold=decision_threshold)
        self._store_init_kwargs(locals())
        self.weight = nn.Parameter(torch.zeros(1))

    def forward_scores(self, td: TensorDict) -> torch.Tensor:
        alpha = torch.sigmoid(self.weight).to(self.device)
        # 1 - vgae_conf = recon_mean/(1+recon_mean): high when VGAE sees high recon error (attack)
        vgae_anom = 1.0 - td["vgae", "conf"].squeeze(-1).to(self.device)
        # GAT attack probability: probs[:,1]
        gat_attack = td["gat", "probs"][..., 1].to(self.device)
        return torch.clamp((1 - alpha) * vgae_anom + alpha * gat_attack, 1e-7, 1 - 1e-7)

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log("alpha", torch.sigmoid(self.weight).item())
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
