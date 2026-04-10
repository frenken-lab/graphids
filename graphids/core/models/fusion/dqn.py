"""Vanilla DQN for dynamic fusion of GAT and VGAE outputs.

Fusion treats each graph as an independent context, so there is no
bootstrapped next state: ``targets = rewards`` and no target network
is needed. For a closed-form alternative, see ``BanditFusionModule``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .base import STATE_DIM, FusionModuleBase, build_mlp_body


class QNetwork(nn.Module):
    """Q-network with configurable depth and width."""

    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.net = nn.Sequential(
            *build_mlp_body(state_dim, hidden_dim, num_layers),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNFusionModule(FusionModuleBase):
    """Vanilla DQN with gradient updates for dynamic fusion."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        alpha_steps: int = 21,
        lr: float = 1e-3,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        buffer_size: int = 50000,
        batch_size: int = 128,
        *,
        hidden_dim: int = 128,
        num_layers: int = 3,
        weight_decay: float = 1e-5,
        decision_threshold: float = 0.5,
        gpu_training_steps: int = 1,
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
        self.hparams = self._capture_hparams(locals())

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gpu_training_steps = gpu_training_steps

        self.q_network = QNetwork(state_dim, alpha_steps, hidden_dim, num_layers)
        self._dqn_optimizer = optim.AdamW(
            self.q_network.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.loss_fn = nn.SmoothL1Loss()

    def build_optimizers(self, max_epochs: int):
        # DQN manages its own optimizer internally.
        # Return it so the trainer can save/restore state.
        return self._dqn_optimizer, None

    # -- Exploration strategy ------------------------------------------------

    def select_action_batch(
        self, states: torch.Tensor, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Epsilon-greedy batch action selection."""
        norm_states = self.reward_calc.normalize(states)

        with torch.no_grad():
            q_values = self.q_network(norm_states.to(self.device))
            greedy_actions = q_values.argmax(dim=1).cpu()

        if training:
            rand_mask = torch.rand(len(states)) < self.epsilon
            random_actions = torch.randint(0, self.alpha_steps, (len(states),))
            actions = torch.where(rand_mask, random_actions, greedy_actions)
        else:
            actions = greedy_actions

        alphas = self.alpha_values[actions]
        return actions, alphas, norm_states

    # -- Learning update -----------------------------------------------------

    def train_episode(self, states: torch.Tensor, labels: torch.Tensor) -> dict:
        actions, alphas, norm_states = self.select_action_batch(states, training=True)
        preds = (alphas > self.decision_threshold).long()
        rewards = self.reward_calc.compute(preds, labels, norm_states, alphas)
        self._buffer.add_batch(norm_states, actions, rewards)

        loss = None
        for _ in range(self.gpu_training_steps):
            loss = self._gradient_step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return {
            "avg_reward": rewards.mean().item(),
            "avg_alpha": alphas.mean().item(),
            "epsilon": self.epsilon,
            "loss": loss,
        }

    # -- DQN-specific internals ----------------------------------------------

    def _gradient_step(self) -> float | None:
        """One DQN gradient step from replay buffer.

        targets = rewards (no bootstrapping — each graph is independent).
        """
        if len(self._buffer) < self.batch_size:
            return None

        states, actions, rewards = self._buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q, rewards)

        self._dqn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self._dqn_optimizer.step()

        return loss.item()

    def q_values(self, norm_states: torch.Tensor) -> torch.Tensor:
        """Q-values for normalized states. Shape: [N, action_dim]."""
        with torch.no_grad():
            return self.q_network(norm_states.to(self.device)).cpu()
