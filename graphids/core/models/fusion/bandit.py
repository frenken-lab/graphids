"""Neural-LinUCB contextual bandit for fusion (Xu et al., ICLR 2022).

Neural backbone learns representations, per-arm ridge regression provides
UCB exploration on the last layer. No target network, no gamma, no sequential
dependency — each graph is an independent context.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import STATE_DIM, FusionModuleBase, build_mlp_body


class Backbone(nn.Module):
    """MLP backbone that outputs representations (no final prediction layer)."""

    def __init__(self, state_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.net = build_mlp_body(state_dim, hidden_dim, num_layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BanditFusionModule(FusionModuleBase):
    """Neural-LinUCB: deep backbone + per-arm linear UCB.

    Action space: K discrete alpha values in [0, 1].
    State: 15-D fusion feature vector (VGAE 8-D + GAT 7-D).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        alpha_steps: int = 21,
        ucb_alpha: float = 1.0,
        lambda_reg: float = 1.0,
        hidden_dim: int = 128,
        num_layers: int = 3,
        backbone_lr: float = 0.001,
        backbone_retrain_freq: int = 50,
        backbone_epochs: int = 5,
        buffer_size: int = 100_000,
        batch_size: int = 128,
        decision_threshold: float = 0.5,
        reward_kwargs: dict | None = None,
    ):
        super().__init__(
            state_dim=state_dim,
            alpha_steps=alpha_steps,
            batch_size=batch_size,
            buffer_size=buffer_size,
            decision_threshold=decision_threshold,
            reward_kwargs=reward_kwargs,
        )
        self.save_hyperparameters()

        self.ucb_alpha = ucb_alpha
        self.lambda_reg = lambda_reg
        self.backbone_retrain_freq = backbone_retrain_freq
        self.backbone_epochs = backbone_epochs

        # Neural backbone
        self.backbone = Backbone(state_dim, hidden_dim, num_layers)
        self.backbone_optimizer = optim.AdamW(self.backbone.parameters(), lr=backbone_lr)
        d = self.backbone.out_dim

        # Per-arm ridge regression (registered buffers — auto-checkpointed)
        self.register_buffer(
            "A_inv",
            torch.eye(d).unsqueeze(0).repeat(alpha_steps, 1, 1) / lambda_reg,
        )
        self.register_buffer("b", torch.zeros(alpha_steps, d))
        self.register_buffer("theta", torch.zeros(alpha_steps, d))

        self._episode = 0
        self._ucb_widths: list[float] = []

    def configure_optimizers(self):
        return self.backbone_optimizer

    # -- Exploration strategy ------------------------------------------------

    def select_action_batch(
        self, states: torch.Tensor, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """UCB-based action selection for a batch of contexts."""
        norm_states = self.reward_calc.normalize(states)

        with torch.no_grad():
            z = self.backbone(norm_states.to(self.device))
            mu = torch.einsum("kd,nd->nk", self.theta, z)

            if training:
                Az = torch.einsum("kij,nj->nki", self.A_inv, z)
                ucb = self.ucb_alpha * torch.sqrt((z.unsqueeze(1) * Az).sum(dim=2).clamp(min=0.0))
                scores = mu + ucb
                self._ucb_widths.append(ucb.mean().item())
            else:
                scores = mu

        actions = scores.argmax(dim=1)
        alphas = self.alpha_values[actions]
        return actions, alphas, norm_states

    # -- Learning update -----------------------------------------------------

    def train_episode(self, states: torch.Tensor, labels: torch.Tensor) -> dict:
        actions, alphas, norm_states = self.select_action_batch(states, training=True)

        anomaly_scores, gat_probs = self.reward_calc.derive_scores(norm_states)
        fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
        preds = (fused_scores > self.decision_threshold).long()

        rewards = self.reward_calc.compute(preds, labels, norm_states, alphas)
        self.update_linear(norm_states, actions, rewards)
        self._buffer.add_batch(norm_states.cpu(), actions.cpu(), rewards.cpu())

        self._episode += 1
        backbone_loss = None
        if self._episode % self.backbone_retrain_freq == 0:
            backbone_loss = self.retrain_backbone()

        correct = (preds == labels).sum().item()
        result = {
            "accuracy": correct / len(labels),
            "avg_reward": rewards.mean().item(),
            "avg_alpha": alphas.mean().item(),
            "alpha_std": alphas.std().item(),
            "avg_ucb_width": float(np.mean(self._ucb_widths[-50:])) if self._ucb_widths else 0.0,
        }
        if backbone_loss is not None:
            result["backbone_loss"] = backbone_loss
        return result

    # -- Bandit-specific internals -------------------------------------------

    def update_linear(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ) -> None:
        """Sherman-Morrison incremental update to per-arm ridge regression."""
        with torch.no_grad():
            z = self.backbone(states.to(self.device))
            for a in range(self.alpha_steps):
                mask = actions == a
                if not mask.any():
                    continue
                z_a = z[mask]
                r_a = rewards[mask].to(self.device)
                for i in range(len(z_a)):
                    zi = z_a[i]
                    Az = self.A_inv[a] @ zi
                    denom = 1.0 + zi @ Az
                    self.A_inv[a] -= torch.outer(Az, Az) / denom
                    self.b[a] += r_a[i] * zi
                self.theta[a] = self.A_inv[a] @ self.b[a]

    def retrain_backbone(self) -> float | None:
        """Retrain backbone on buffered experiences. Returns avg loss or None."""
        if len(self._buffer) < self.batch_size:
            return None

        was_training = self.backbone.training
        self.backbone.train()
        total_loss = 0.0
        for _ in range(self.backbone_epochs):
            states, actions, rewards = self._buffer.sample(self.batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)

            z = self.backbone(states)
            preds = (self.theta[actions] * z).sum(dim=1)
            loss = nn.functional.mse_loss(preds, rewards)

            self.backbone_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
            self.backbone_optimizer.step()
            total_loss += loss.item()

        self.backbone.train(was_training)

        d = self.backbone.out_dim
        self.A_inv.copy_(
            torch.eye(d, device=self.A_inv.device).unsqueeze(0).repeat(self.alpha_steps, 1, 1)
            / self.lambda_reg
        )
        self.b.zero_()
        self.theta.zero_()

        return total_loss / self.backbone_epochs

    def q_values(self, norm_states: torch.Tensor) -> torch.Tensor:
        """Per-arm expected rewards for normalized states. Shape: [N, K]."""
        with torch.no_grad():
            z = self.backbone(norm_states.to(self.device))
            return torch.einsum("kd,nd->nk", self.theta, z).cpu()
