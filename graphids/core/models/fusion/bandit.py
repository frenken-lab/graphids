"""Neural-LinUCB contextual bandit (Xu et al., ICLR 2022).

Uses torchrl's ``LossModule`` for the backbone-MSE refit so the
sample → loss → step skeleton is shared with DQN through ``RLFusionBase``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from torchrl.objectives import LossModule

from .base import RLFusionBase, build_mlp_body


class _Backbone(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.net = build_mlp_body(state_dim, hidden_dim, num_layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LinUCBLoss(LossModule):
    """torchrl LossModule: MSE between predicted reward (z·θ_a) and stored
    reward. ``theta`` is a buffer on the parent module — we hold a Python
    reference (not registered) so the optimizer only updates the backbone."""

    def __init__(self, backbone: _Backbone, theta: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        # Plain attribute — nn.Module.__setattr__ stores non-Parameter tensors
        # without registering as a buffer, so the optimizer only sees backbone params.
        self._theta = theta

    def forward(self, td: TensorDict) -> TensorDict:
        states = td["observation"]
        actions = td["action"]
        rewards = td["next", "reward"].squeeze(-1)
        z = self.backbone(states)
        preds = (self._theta[actions] * z).sum(dim=1)
        loss = nn.functional.mse_loss(preds, rewards)
        return TensorDict({"loss": loss}, batch_size=[])


class BanditFusionModule(RLFusionBase):
    """Neural-LinUCB: backbone + per-arm ridge with Sherman-Morrison online
    updates, and a frequency-gated backbone refit (via torchrl LossModule)."""

    def __init__(
        self,
        state_dim: int = 15,
        alpha_steps: int = 21,
        ucb_alpha: float = 1.0,
        lambda_reg: float = 1.0,
        hidden_dim: int = 128,
        num_layers: int = 3,
        backbone_lr: float = 1e-3,
        backbone_retrain_freq: int = 50,
        backbone_epochs: int = 5,
        buffer_size: int = 100_000,
        batch_size: int = 128,
        decision_threshold: float = 0.5,
        reward_kwargs: dict | None = None,
    ):
        super().__init__(
            buffer_size=buffer_size,
            batch_size=batch_size,
            state_dim=state_dim,
            alpha_steps=alpha_steps,
            decision_threshold=decision_threshold,
            reward_kwargs=reward_kwargs,
        )
        self._store_init_kwargs(locals())

        self.backbone = _Backbone(state_dim, hidden_dim, num_layers)
        d = self.backbone.out_dim

        self.register_buffer(
            "A_inv", torch.eye(d).unsqueeze(0).repeat(alpha_steps, 1, 1) / lambda_reg
        )
        self.register_buffer("b", torch.zeros(alpha_steps, d))
        self.register_buffer("theta", torch.zeros(alpha_steps, d))

        # torchrl LossModule wired to the same backbone + theta buffer.
        self.loss_module = _LinUCBLoss(self.backbone, self.theta)

        # gpu_training_steps drives RLFusionBase._learn_step's inner loop.
        self.gpu_training_steps = backbone_epochs

        self._optimizer = optim.AdamW(self.backbone.parameters(), lr=backbone_lr)
        self._episode = 0
        self._ucb_widths: list[float] = []

    # -- RLFusionBase hooks --------------------------------------------------

    def _score_actions(self, td: TensorDict, training: bool) -> None:
        states = td["observation"]
        z = self.backbone(states)
        mu = torch.einsum("kd,nd->nk", self.theta, z)
        if training and self.ucb_alpha > 0:
            Az = torch.einsum("kij,nj->nki", self.A_inv, z)
            ucb = self.ucb_alpha * torch.sqrt(
                (z.unsqueeze(1) * Az).sum(dim=2).clamp(min=0.0)
            )
            scores = mu + ucb
            self._ucb_widths.append(ucb.mean().item())
        else:
            scores = mu
        td["action"] = scores.argmax(dim=1)

    def _after_act(self, actions, norm_states, rewards) -> None:
        self._update_linear(norm_states.to(self.device), actions, rewards)
        self._episode += 1

    def _should_learn(self) -> bool:
        return self._episode % self.backbone_retrain_freq == 0

    def _after_learn(self) -> None:
        # Reset ridge state after each backbone refit (theta is now stale w.r.t. new z).
        d = self.backbone.out_dim
        self.A_inv.copy_(
            torch.eye(d, device=self.A_inv.device).unsqueeze(0).repeat(self.alpha_steps, 1, 1)
            / self.lambda_reg
        )
        self.b.zero_()
        self.theta.zero_()

    def _extra_metrics(self) -> dict:
        return {
            "avg_ucb_width": float(np.mean(self._ucb_widths[-50:]))
            if self._ucb_widths
            else 0.0,
        }

    # -- Sherman-Morrison ----------------------------------------------------

    def _update_linear(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ) -> None:
        with torch.no_grad():
            z = self.backbone(states)
            for a in range(self.alpha_steps):
                mask = actions == a
                if not mask.any():
                    continue
                z_a = z[mask]
                r_a = rewards[mask].to(self.device)
                for i in range(len(z_a)):
                    zi = z_a[i]
                    Az = self.A_inv[a] @ zi
                    self.A_inv[a] -= torch.outer(Az, Az) / (1.0 + zi @ Az)
                    self.b[a] += r_a[i] * zi
                self.theta[a] = self.A_inv[a] @ self.b[a]

    def q_values(self, norm_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.backbone(norm_states.to(self.device))
            return torch.einsum("kd,nd->nk", self.theta, z).cpu()
