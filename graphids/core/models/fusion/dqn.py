from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from graphids.log import get_logger

from ._nn import TensorReplayBuffer, build_mlp_body
from .fusion_baselines import FusionModuleBase
from .fusion_features import STATE_DIM
from .fusion_reward import FusionRewardCalculator

log = get_logger(__name__)


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


# ---------------------------------------------------------------------------
# DQN Fusion Agent (vectorized training)
# See ~/plans/fusion-redesign.md for RL vs supervised analysis
# ---------------------------------------------------------------------------


class DQNFusionModule(FusionModuleBase):
    """Vanilla DQN with gradient updates for dynamic fusion of GAT and VGAE outputs.

    Fusion treats each graph as an independent context, so there is no
    bootstrapped next state: ``targets = rewards`` and no target network
    is needed. For a closed-form alternative, see ``BanditFusionModule``.
    """

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
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.register_buffer("_alpha_values_t", torch.linspace(0, 1, alpha_steps))
        self.action_dim = alpha_steps
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.decision_threshold = decision_threshold
        self.gpu_training_steps = gpu_training_steps

        # Network (nn.Module child — Lightning auto-transfers and checkpoints)
        self.q_network = QNetwork(state_dim, alpha_steps, hidden_dim, num_layers)

        # Optimizer and loss
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.SmoothL1Loss()

        self._buffer = TensorReplayBuffer(buffer_size, state_dim)
        self.reward_calc = FusionRewardCalculator(**(reward_kwargs or {}))

        log.info("dqn_agent_initialized", actions=alpha_steps, state_dim=state_dim)

    def configure_optimizers(self):
        return self.optimizer

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
    # Action selection
    # ------------------------------------------------------------------

    def select_action_batch(
        self, states: torch.Tensor, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Epsilon-greedy batch action selection. Returns (actions, alphas, norm_states)."""
        norm_states = self.reward_calc.normalize(states)

        with torch.no_grad():
            q_values = self.q_network(norm_states.to(self.device))
            greedy_actions = q_values.argmax(dim=1).cpu()

        if training:
            rand_mask = torch.rand(len(states)) < self.epsilon
            random_actions = torch.randint(0, self.action_dim, (len(states),))
            actions = torch.where(rand_mask, random_actions, greedy_actions)
        else:
            actions = greedy_actions

        alphas = self._alpha_values_t[actions]
        return actions, alphas, norm_states

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
        """One DQN gradient step from replay buffer.

        Fusion has no real next-state: targets = rewards (pure reward
        regression, no bootstrapping), so no target network is needed.
        """
        if len(self._buffer) < self.batch_size:
            return None

        states, actions, rewards = self._buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q, rewards)

        opt = self.optimizer
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        opt.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Training episode (called from fusion pipeline)
    # ------------------------------------------------------------------

    def train_episode(
        self, states: torch.Tensor, labels: torch.Tensor
    ) -> dict:
        """One training episode: select → reward → replay → gradient steps → epsilon decay.

        Args:
            states: [N, D] raw state features
            labels: [N] ground truth labels

        Returns:
            Dict with avg_reward, avg_alpha, epsilon, loss.
        """
        actions, alphas, norm_states = self.select_action_batch(states, training=True)
        preds = (alphas > self.decision_threshold).long()
        rewards = self.reward_calc.compute(preds, labels, norm_states, alphas)
        self._buffer.add_batch(norm_states, actions, rewards)

        loss = None
        for _ in range(self.gpu_training_steps):
            loss = self.train_step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return {
            "avg_reward": rewards.mean().item(),
            "avg_alpha": alphas.mean().item(),
            "epsilon": self.epsilon,
            "loss": loss,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, states: torch.Tensor) -> dict:
        """Greedy fused prediction (no exploration)."""
        from .fusion_reward import fused_predict
        return fused_predict(self, states)

    def q_values(self, norm_states: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for normalized states. Shape: [N, action_dim]."""
        with torch.no_grad():
            return self.q_network(norm_states.to(self.device)).cpu()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_batch(self, states: torch.Tensor, labels: torch.Tensor) -> dict:
        """Greedy evaluation with proper fused prediction."""
        was_training = self.q_network.training
        self.q_network.eval()

        result = self.predict(states)
        preds, norm_states, alphas = result["preds"], result["norm_states"], result["alphas"]

        correct = (preds == labels).sum().item()
        rewards = self.reward_calc.compute(preds, labels, norm_states, alphas)

        if was_training:
            self.q_network.train()

        return {
            "accuracy": correct / len(labels),
            "avg_reward": rewards.mean().item(),
            "avg_alpha": alphas.mean().item(),
            "alpha_std": alphas.std().item(),
        }
