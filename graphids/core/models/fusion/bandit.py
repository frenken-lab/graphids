"""Neural-LinUCB contextual bandit for fusion (Xu et al., ICLR 2022).

Neural backbone learns representations, per-arm ridge regression provides
UCB exploration on the last layer. No target network, no gamma, no sequential
dependency — each graph is an independent context.

Reference: "Neural Contextual Bandits with UCB-based Exploration" (Zhou et al., 2020)
           "Neural Contextual Bandits with Deep Representation and Shallow Exploration" (Xu et al., 2022)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from graphids.log import get_logger

from ._nn import TensorReplayBuffer, build_mlp_body
from .fusion_baselines import FusionModuleBase
from .fusion_features import fusion_state_dim
from .fusion_reward import FusionRewardCalculator

log = get_logger(__name__)

_DEFAULT_STATE_DIM = fusion_state_dim()


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
        state_dim: int = _DEFAULT_STATE_DIM,
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
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.alpha_steps = alpha_steps
        self.register_buffer("alpha_values", torch.linspace(0, 1, alpha_steps))
        self.ucb_alpha = ucb_alpha
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.backbone_retrain_freq = backbone_retrain_freq
        self.backbone_epochs = backbone_epochs
        self.decision_threshold = decision_threshold

        # Neural backbone (nn.Module child — auto-transferred and checkpointed)
        self.backbone = Backbone(state_dim, hidden_dim, num_layers)
        self.backbone_optimizer = optim.AdamW(self.backbone.parameters(), lr=backbone_lr)
        d = self.backbone.out_dim

        # Per-arm ridge regression as registered buffers (auto-checkpointed, auto device transfer)
        self.register_buffer(
            "A_inv",
            torch.eye(d).unsqueeze(0).repeat(alpha_steps, 1, 1) / lambda_reg,
        )
        self.register_buffer("b", torch.zeros(alpha_steps, d))
        self.register_buffer("theta", torch.zeros(alpha_steps, d))

        # Replay buffer for backbone retraining
        self._buffer = TensorReplayBuffer(buffer_size, state_dim)

        # Reward calculator (shared with DQN)
        self.reward_calc = FusionRewardCalculator(**(reward_kwargs or {}))

        # Tracking
        self.state_dim = state_dim
        self._total_reward = 0.0
        self._total_steps = 0
        self._episode = 0
        self._ucb_widths: list[float] = []

        log.info("bandit_initialized", arms=alpha_steps, backbone_dim=d, ucb_alpha=ucb_alpha)

    def configure_optimizers(self):
        return self.backbone_optimizer

    # ------------------------------------------------------------------
    # Lightning training entry point
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        states, labels = batch
        result = self.train_episode(states, labels)
        for k, v in result.items():
            if v is not None:
                self.log(k, float(v), prog_bar=(k in ("avg_reward", "accuracy")))

    # ------------------------------------------------------------------
    # Action selection (batch)
    # ------------------------------------------------------------------

    def select_action_batch(
        self, states: torch.Tensor, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """UCB-based action selection for a batch of contexts.

        Returns:
            actions: [N] long, alphas: [N] float, norm_states: [N, D] float
        """
        norm_states = self.reward_calc.normalize(states)

        with torch.no_grad():
            z = self.backbone(norm_states.to(self.device))  # [N, d]

            # Predicted reward per arm: theta_a^T z  → [N, K]
            mu = torch.einsum("kd,nd->nk", self.theta, z)

            if training:
                # UCB bonus: alpha * sqrt(z^T A_inv_a z) → [N, K]
                # For each arm k: z @ A_inv[k] @ z^T → scalar per (n, k)
                Az = torch.einsum("kij,nj->nki", self.A_inv, z)  # [N, K, d]
                ucb = self.ucb_alpha * torch.sqrt(
                    (z.unsqueeze(1) * Az).sum(dim=2).clamp(min=0.0)
                )  # [N, K]
                scores = mu + ucb
                self._ucb_widths.append(ucb.mean().item())
            else:
                scores = mu

        actions = scores.argmax(dim=1)
        alphas = self.alpha_values[actions]
        return actions, alphas, norm_states

    # ------------------------------------------------------------------
    # Linear update (closed-form, no gradient)
    # ------------------------------------------------------------------

    def update_linear(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ) -> None:
        """Sherman-Morrison incremental update to per-arm ridge regression.

        For each arm a that was selected, updates:
            A_inv_a ← A_inv_a - (A_inv_a z z^T A_inv_a) / (1 + z^T A_inv_a z)
            b_a ← b_a + r * z
            theta_a ← A_inv_a @ b_a
        """
        with torch.no_grad():
            z = self.backbone(states.to(self.device))  # [N, d]

            for a in range(self.alpha_steps):
                mask = actions == a
                if not mask.any():
                    continue
                z_a = z[mask]  # [n_a, d]
                r_a = rewards[mask].to(self.device)  # [n_a]

                # Batch Sherman-Morrison: process all samples for this arm
                for i in range(len(z_a)):
                    zi = z_a[i]  # [d]
                    Az = self.A_inv[a] @ zi  # [d]
                    denom = 1.0 + zi @ Az
                    self.A_inv[a] -= torch.outer(Az, Az) / denom
                    self.b[a] += r_a[i] * zi

                self.theta[a] = self.A_inv[a] @ self.b[a]

    # ------------------------------------------------------------------
    # Backbone retraining (periodic, from buffer)
    # ------------------------------------------------------------------

    def _store_buffer(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ) -> None:
        """Store experiences in replay buffer for backbone retraining."""
        self._buffer.add_batch(states.cpu(), actions.cpu(), rewards.cpu())

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

            z = self.backbone(states)  # [B, d]
            # Predicted reward for the taken action: theta_a^T z
            preds = (self.theta[actions] * z).sum(dim=1)
            loss = nn.functional.mse_loss(preds, rewards)

            self.backbone_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
            self.backbone_optimizer.step()
            total_loss += loss.item()

        self.backbone.train(was_training)

        # Reset linear models after backbone change (representations shifted)
        # Use in-place ops to preserve registered buffer registration
        d = self.backbone.out_dim
        self.A_inv.copy_(
            torch.eye(d, device=self.A_inv.device).unsqueeze(0).repeat(
                self.alpha_steps, 1, 1
            ) / self.lambda_reg
        )
        self.b.zero_()
        self.theta.zero_()

        return total_loss / self.backbone_epochs

    # ------------------------------------------------------------------
    # Training episode (called from fusion pipeline)
    # ------------------------------------------------------------------

    def train_episode(
        self, states: torch.Tensor, labels: torch.Tensor
    ) -> dict:
        """One training episode: select actions, compute rewards, update.

        Args:
            states: [N, D] raw state features
            labels: [N] ground truth labels

        Returns:
            Dict with avg_reward, avg_alpha, accuracy.
        """
        actions, alphas, norm_states = self.select_action_batch(states, training=True)

        # Fused prediction
        anomaly_scores, gat_probs = self.reward_calc.derive_scores(norm_states)
        fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
        preds = (fused_scores > self.decision_threshold).long()

        # Reward
        rewards = self.reward_calc.compute(preds, labels, norm_states, alphas)

        # Update linear models (closed-form)
        self.update_linear(norm_states, actions, rewards)

        # Store for backbone retraining
        self._store_buffer(norm_states, actions, rewards)

        # Periodic backbone retrain
        self._episode += 1
        backbone_loss = None
        if self._episode % self.backbone_retrain_freq == 0:
            backbone_loss = self.retrain_backbone()

        # Track
        self._total_reward += rewards.sum().item()
        self._total_steps += len(states)

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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, states: torch.Tensor) -> dict:
        """Greedy fused prediction (no UCB exploration)."""
        from .fusion_reward import fused_predict
        return fused_predict(self, states)

    def q_values(self, norm_states: torch.Tensor) -> torch.Tensor:
        """Compute per-arm expected rewards for normalized states. Shape: [N, K]."""
        with torch.no_grad():
            z = self.backbone(norm_states.to(self.device))
            return torch.einsum("kd,nd->nk", self.theta, z).cpu()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_batch(self, states: torch.Tensor, labels: torch.Tensor) -> dict:
        """Greedy evaluation (no UCB exploration)."""
        result = self.predict(states)
        preds, norm_states, alphas = result["preds"], result["norm_states"], result["alphas"]

        correct = (preds == labels).sum().item()
        rewards = self.reward_calc.compute(preds, labels, norm_states, alphas)

        return {
            "accuracy": correct / len(labels),
            "avg_reward": rewards.mean().item(),
            "avg_alpha": alphas.mean().item(),
            "alpha_std": alphas.std().item(),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def regret_stats(self) -> dict:
        """Return cumulative regret estimate and UCB width trend."""
        return {
            "cumulative_reward": self._total_reward,
            "total_steps": self._total_steps,
            "avg_reward_per_step": self._total_reward / max(self._total_steps, 1),
            "avg_ucb_width": float(np.mean(self._ucb_widths[-50:])) if self._ucb_widths else 0.0,
            "episodes": self._episode,
        }
