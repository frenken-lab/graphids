from __future__ import annotations

import structlog
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .fusion_reward import FusionRewardCalculator

log = structlog.get_logger()


class QNetwork(nn.Module):
    """Q-network with configurable depth and width."""

    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, cfg) -> "QNetwork":
        """Construct from a config."""
        from .registry import fusion_state_dim

        return cls(
            state_dim=fusion_state_dim(),
            action_dim=cfg.fusion.alpha_steps,
            hidden_dim=cfg.dqn.hidden,
            num_layers=cfg.dqn.layers,
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Tensor Replay Buffer (replaces deque of tuples)
# ---------------------------------------------------------------------------


class TensorReplayBuffer:
    """Fixed-size circular buffer backed by contiguous tensors.

    Stores (state, action, reward) triples only — next_state is always
    identical to state in the current fusion formulation (see TODO below).
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.states = torch.zeros(capacity, state_dim)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self._pos = 0
        self._size = 0

    def add_batch(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        """Add a batch of experiences. Wraps around when full."""
        n = len(states)
        if n >= self.capacity:
            # Keep only the last `capacity` items
            states = states[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            n = self.capacity

        end = self._pos + n
        if end <= self.capacity:
            self.states[self._pos : end] = states
            self.actions[self._pos : end] = actions
            self.rewards[self._pos : end] = rewards
        else:
            first = self.capacity - self._pos
            self.states[self._pos :] = states[:first]
            self.actions[self._pos :] = actions[:first]
            self.rewards[self._pos :] = rewards[:first]
            rest = n - first
            self.states[:rest] = states[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]

        self._pos = (self._pos + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random sample without replacement (or with, if batch_size > size)."""
        idx = torch.randint(0, self._size, (batch_size,))
        return self.states[idx], self.actions[idx], self.rewards[idx]

    def __len__(self):
        return self._size


# ---------------------------------------------------------------------------
# DQN Fusion Agent (vectorized training)
# See ~/plans/fusion-redesign.md for RL vs supervised analysis
# ---------------------------------------------------------------------------


class EnhancedDQNFusionAgent:
    """DQN agent for dynamic fusion of GAT and VGAE outputs.

    Designed for streaming/temporal scenarios where sequential decisions matter
    (gamma > 0, target network, Double DQN). For independent graphs, prefer
    the contextual bandit in bandit.py.
    """

    def __init__(
        self,
        alpha_steps=21,
        lr=1e-3,
        gamma=0.0,
        epsilon=0.2,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        buffer_size=50000,
        batch_size=128,
        target_update_freq=100,
        device="cpu",
        *,
        state_dim,
        hidden_dim=128,
        num_layers=3,
        weight_decay=1e-5,
        scheduler_patience=1000,
        decision_threshold: float = 0.5,
        reward_kwargs: dict | None = None,
    ):
        self._alpha_values_t = torch.linspace(0, 1, alpha_steps)
        self.action_dim = alpha_steps
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.decision_threshold = decision_threshold

        # Networks
        self.q_network = QNetwork(state_dim, alpha_steps, hidden_dim, num_layers).to(device)
        self.target_network = QNetwork(state_dim, alpha_steps, hidden_dim, num_layers).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer and loss
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=scheduler_patience, factor=0.8
        )
        self.loss_fn = nn.SmoothL1Loss()

        self._buffer = TensorReplayBuffer(buffer_size, state_dim)
        self.reward_calc = FusionRewardCalculator(**(reward_kwargs or {}))
        self.training_step = 0
        self.update_counter = 0

        log.info("dqn_agent_initialized", actions=alpha_steps, state_dim=state_dim)

    @classmethod
    def from_config(
        cls, cfg, device: str = "cpu", *, inference: bool = False,
    ) -> EnhancedDQNFusionAgent:
        """Create agent from config. Set inference=True for eval/serve (no exploration)."""
        from .registry import fusion_state_dim

        reward_kwargs = dict(
            vgae_weights=list(cfg.dqn.vgae_error_weights),
            reward_correct=cfg.dqn.reward_correct,
            reward_incorrect=cfg.dqn.reward_incorrect,
            confidence_weight=cfg.dqn.confidence_weight,
            combined_conf_weight=cfg.dqn.combined_conf_weight,
            disagreement_penalty=cfg.dqn.disagreement_penalty,
            overconf_penalty=cfg.dqn.overconf_penalty,
            balance_weight=cfg.dqn.balance_weight,
        )
        kwargs = dict(
            lr=cfg.fusion.lr,
            gamma=cfg.dqn.gamma,
            buffer_size=cfg.dqn.buffer_size,
            batch_size=cfg.dqn.batch_size,
            target_update_freq=cfg.dqn.target_update,
            device=device,
            state_dim=fusion_state_dim(),
            alpha_steps=cfg.fusion.alpha_steps,
            hidden_dim=cfg.dqn.hidden,
            num_layers=cfg.dqn.layers,
            weight_decay=cfg.dqn.weight_decay,
            scheduler_patience=cfg.dqn.scheduler_patience,
            decision_threshold=cfg.fusion.decision_threshold,
            reward_kwargs=reward_kwargs,
        )
        if inference:
            kwargs.update(epsilon=0.0, epsilon_decay=1.0, min_epsilon=0.0)
        else:
            kwargs.update(
                epsilon=cfg.dqn.epsilon,
                epsilon_decay=cfg.dqn.epsilon_decay,
                min_epsilon=cfg.dqn.min_epsilon,
            )
        return cls(**kwargs)

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

    def store_experiences_batch(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ):
        """Store a batch of experiences in the tensor replay buffer."""
        self._buffer.add_batch(states, actions, rewards)

    def train_step(self) -> float | None:
        """One Double DQN gradient step from replay buffer."""
        if len(self._buffer) < self.batch_size:
            return None

        states, actions, rewards = self._buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # No real next-state in fusion formulation — gamma=0 makes this
            # pure reward maximization (no bootstrapping from same state).
            targets = rewards

        loss = self.loss_fn(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.training_step += 1
        return loss.item()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_batch(self, states: torch.Tensor, labels: torch.Tensor) -> dict:
        """Greedy evaluation with proper fused prediction."""
        was_training = self.q_network.training
        self.q_network.eval()

        actions, alphas, norm_states = self.select_action_batch(states, training=False)

        anomaly_scores, gat_probs = self.reward_calc.derive_scores(norm_states)
        fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
        preds = (fused_scores > self.decision_threshold).long()

        correct = (preds == labels).sum().item()
        rewards = self.reward_calc.compute(preds, labels, norm_states, alphas)

        if was_training:
            self.q_network.train()

        result = {
            "accuracy": correct / len(labels),
            "avg_reward": rewards.mean().item(),
            "avg_alpha": alphas.mean().item(),
            "alpha_std": alphas.std().item(),
        }
        self.scheduler.step(result["avg_reward"])
        return result

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_or_path: dict | str | Path) -> None:
        """Load Q-network and target network weights."""
        if isinstance(checkpoint_or_path, dict):
            sd = checkpoint_or_path
        else:
            sd = torch.load(checkpoint_or_path, map_location="cpu", weights_only=True)
        self.q_network.load_state_dict(sd["q_network"])
        self.target_network.load_state_dict(sd["target_network"])
        if "epsilon" in sd:
            self.epsilon = sd["epsilon"]

    @property
    def buffer_size_current(self) -> int:
        """Current number of experiences in the replay buffer."""
        return len(self._buffer)
