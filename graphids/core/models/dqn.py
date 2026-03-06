import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fusion Agent ABC
# ---------------------------------------------------------------------------


class FusionAgent(ABC):
    """Abstract base class for fusion agents.

    All fusion agents operate on the same N-D state vector produced by
    the model registry's extractors (VGAE 8-D + GAT 7-D = 15-D).
    """

    @abstractmethod
    def train_on_cache(
        self,
        train_states: torch.Tensor,
        train_labels: torch.Tensor,
        val_states: torch.Tensor,
        val_labels: torch.Tensor,
        cfg,
    ) -> float:
        """Train the agent on cached predictions. Returns best validation accuracy."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Return serializable state dict for checkpointing."""

    @abstractmethod
    def fuse(self, state_features: np.ndarray) -> int:
        """Given a state vector, return a fused binary prediction (0 or 1)."""


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
        """Construct from a PipelineConfig."""
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
# ---------------------------------------------------------------------------
# TODO(open-question): Is RL the right formulation for fusion?
#
# The current setup is a contextual bandit, NOT a sequential MDP:
#   - next_state == state (no state transitions)
#   - done == False always (no terminal states)
#   - Samples are i.i.d. from a pre-cached dataset (no environment dynamics)
#
# The Bellman target r + gamma * max Q(s', a') degenerates into
# r + gamma * max Q(s, a'), a self-referential loop. The discount factor
# just inflates Q-values without adding information.
#
# Alternatives to evaluate (compare F1/accuracy on held-out data):
#   1. gamma=0 (proper bandit): target = r, no target network needed
#   2. MLPFusionAgent (already implemented): supervised BCE, trains in seconds
#   3. WeightedAvgFusionAgent: single learned alpha, simplest baseline
#   4. Contextual Thompson Sampling or Neural UCB for principled exploration
#
# If MLP matches or beats DQN F1, the RL framing should be dropped entirely.
# See also: WeightedAvgFusionAgent docstring ("if this matches DQN's F1,
# the RL approach is unjustified").
#
# Additional issue: during training, prediction = (alpha > 0.5), but during
# validation, prediction = (fused_score > 0.5). These differ because alpha
# is a fusion weight, not a score. The training reward signal is based on
# a semantically wrong "prediction". This is preserved for compatibility
# but should be fixed if the DQN path is kept.
# ---------------------------------------------------------------------------


class EnhancedDQNFusionAgent:
    """Enhanced DQN agent for dynamic fusion of GAT and VGAE outputs.

    Vectorized training: batch forward passes, batch reward computation,
    tensor replay buffer. Single-sample methods kept for inference.
    """

    def __init__(
        self,
        alpha_steps=21,
        lr=1e-3,
        gamma=0.9,
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
    ):
        # Action and state space
        self.alpha_values = np.linspace(0, 1, alpha_steps)
        self._alpha_values_t = torch.tensor(self.alpha_values, dtype=torch.float32)
        self.action_dim = alpha_steps
        self.state_dim = state_dim

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.buffer_size = buffer_size

        # Networks
        self.q_network = QNetwork(state_dim, self.action_dim, hidden_dim, num_layers).to(
            self.device
        )
        self.target_network = QNetwork(state_dim, self.action_dim, hidden_dim, num_layers).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer and loss
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=1000, factor=0.8
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability

        # Tensor replay buffer (replaces deque of tuples)
        self._buffer = TensorReplayBuffer(buffer_size, state_dim)

        # Training tracking
        self.training_step = 0
        self.update_counter = 0
        self.reward_history: list[float] = []
        self.loss_history: list[float] = []

        # Validation tracking
        self.validation_scores: list[dict] = []
        self.best_validation_score = -float("inf")
        self.patience_counter = 0
        self.max_patience = 5000

        # Derive feature indices from registry (no hardcoded offsets)
        from .registry import feature_layout

        layout = feature_layout()
        vgae_start, vgae_dim, _ = layout["vgae"]
        gat_start, gat_dim, _ = layout["gat"]
        self._confidence_indices = [layout[n][2] for n in layout]
        self._vgae_error_slice = slice(vgae_start, vgae_start + 3)
        self._gat_logit_slice = slice(gat_start, gat_start + 2)
        self._vgae_conf_idx = layout["vgae"][2]
        self._gat_conf_idx = layout["gat"][2]

        # Weights for VGAE anomaly score (used in batch reward computation)
        self._vgae_weights = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float32)

        log.info("DQN Agent initialized: %d actions, state_dim=%d", alpha_steps, self.state_dim)

    # ------------------------------------------------------------------
    # Single-sample methods (inference / serve.py)
    # ------------------------------------------------------------------

    def normalize_state(self, state_features: np.ndarray) -> np.ndarray:
        """Normalize a single state (numpy). Used for inference."""
        if not isinstance(state_features, np.ndarray):
            state_features = np.array(state_features, dtype=np.float32)
        if len(state_features) != self.state_dim:
            raise ValueError(f"Expected {self.state_dim}D state, got {len(state_features)}D")
        state_features = state_features.copy()
        for idx in self._confidence_indices:
            state_features[idx] = np.clip(state_features[idx], 0.0, 1.0)
        return state_features.astype(np.float32)

    def select_action(
        self, state_features: np.ndarray, training: bool = True
    ) -> tuple[float, int, np.ndarray]:
        """Select action for a single state (numpy). Used for inference."""
        state = self.normalize_state(state_features)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if training and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = torch.argmax(q_values).item()

        alpha_value = self.alpha_values[action_idx]
        return alpha_value, action_idx, state

    def _derive_scores(self, state_features: np.ndarray) -> tuple[float, float]:
        """Derive anomaly_score and gat_prob from a single state (numpy)."""
        vgae_errors = state_features[self._vgae_error_slice]
        vgae_weights = np.array([0.4, 0.35, 0.25])
        anomaly_score = float(np.clip(np.sum(vgae_errors * vgae_weights), 0.0, 1.0))

        gat_logits = state_features[self._gat_logit_slice]
        shifted = gat_logits - np.max(gat_logits)
        gat_probs = np.exp(shifted) / np.sum(np.exp(shifted))
        gat_prob = float(gat_probs[1])

        return anomaly_score, gat_prob

    def compute_fusion_reward(
        self, prediction: int, true_label: int, state_features: np.ndarray, alpha: float
    ) -> float:
        """Compute reward for a single sample (numpy). Used for inference/analysis."""
        anomaly_score, gat_prob = self._derive_scores(state_features)

        vgae_confidence = float(state_features[self._vgae_conf_idx])
        gat_confidence = float(state_features[self._gat_conf_idx])
        combined_confidence = max(vgae_confidence, gat_confidence)

        base_reward = 3.0 if prediction == true_label else -3.0
        model_agreement = 1.0 - abs(anomaly_score - gat_prob)

        if prediction == true_label:
            agreement_bonus = 1.0 * model_agreement
            if true_label == 1:
                confidence = max(anomaly_score, gat_prob)
            else:
                confidence = 1.0 - max(anomaly_score, gat_prob)
            confidence_bonus = 0.5 * confidence + 0.3 * combined_confidence
            total_reward = base_reward + agreement_bonus + confidence_bonus
        else:
            disagreement_penalty = -1.0 * (1.0 - model_agreement)
            fused_confidence = alpha * gat_prob + (1 - alpha) * anomaly_score
            if prediction == 1:
                overconf_penalty = -1.5 * fused_confidence
            else:
                overconf_penalty = -1.5 * (1.0 - fused_confidence)
            total_reward = base_reward + disagreement_penalty + overconf_penalty

        balance_bonus = 0.3 * (1.0 - abs(alpha - 0.5) * 2)
        return total_reward + balance_bonus

    # ------------------------------------------------------------------
    # Batch methods (training)
    # ------------------------------------------------------------------

    def _normalize_batch(self, states: torch.Tensor) -> torch.Tensor:
        """Normalize a batch of states. states: [N, D] -> [N, D] float32."""
        states = states.clone().float()
        for idx in self._confidence_indices:
            states[:, idx].clamp_(0.0, 1.0)
        return states

    def select_action_batch(
        self, states: torch.Tensor, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch action selection via epsilon-greedy.

        Args:
            states: [N, D] tensor (CPU)
            training: use epsilon-greedy if True

        Returns:
            actions: [N] long tensor (CPU)
            alphas: [N] float tensor (CPU)
            norm_states: [N, D] float tensor (CPU)
        """
        norm_states = self._normalize_batch(states)

        with torch.no_grad():
            q_values = self.q_network(norm_states.to(self.device))  # [N, action_dim]
            greedy_actions = q_values.argmax(dim=1).cpu()  # [N]

        if training:
            rand_mask = torch.rand(len(states)) < self.epsilon
            random_actions = torch.randint(0, self.action_dim, (len(states),))
            actions = torch.where(rand_mask, random_actions, greedy_actions)
        else:
            actions = greedy_actions

        alphas = self._alpha_values_t[actions]
        return actions, alphas, norm_states

    def _derive_scores_batch(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch derive anomaly_score and gat_prob. states: [N, D] tensor."""
        vgae_errors = states[:, self._vgae_error_slice]  # [N, 3]
        anomaly_scores = (vgae_errors * self._vgae_weights).sum(dim=1).clamp(0.0, 1.0)

        gat_logits = states[:, self._gat_logit_slice]  # [N, 2]
        gat_probs = torch.softmax(gat_logits, dim=1)[:, 1]  # attack class prob

        return anomaly_scores, gat_probs

    def compute_fusion_reward_batch(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        states: torch.Tensor,
        alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized reward computation. All inputs are [N] or [N, D] tensors.

        Returns [N] float tensor of rewards.
        """
        anomaly_scores, gat_probs = self._derive_scores_batch(states)
        vgae_conf = states[:, self._vgae_conf_idx]
        gat_conf = states[:, self._gat_conf_idx]
        combined_conf = torch.max(vgae_conf, gat_conf)

        correct = preds == labels
        base_reward = torch.where(correct, 3.0, -3.0)
        model_agreement = 1.0 - (anomaly_scores - gat_probs).abs()

        # Correct-prediction path
        agreement_bonus = model_agreement
        max_score = torch.max(anomaly_scores, gat_probs)
        confidence = torch.where(labels == 1, max_score, 1.0 - max_score)
        confidence_bonus = 0.5 * confidence + 0.3 * combined_conf
        correct_reward = base_reward + agreement_bonus + confidence_bonus

        # Wrong-prediction path
        disagreement_penalty = -1.0 * (1.0 - model_agreement)
        fused_confidence = alphas * gat_probs + (1 - alphas) * anomaly_scores
        overconf_penalty = torch.where(
            preds == 1,
            -1.5 * fused_confidence,
            -1.5 * (1.0 - fused_confidence),
        )
        wrong_reward = base_reward + disagreement_penalty + overconf_penalty

        total_reward = torch.where(correct, correct_reward, wrong_reward)
        balance_bonus = 0.3 * (1.0 - (alphas - 0.5).abs() * 2)
        return total_reward + balance_bonus

    def store_experiences_batch(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ):
        """Store a batch of experiences in the tensor replay buffer."""
        self._buffer.add_batch(states, actions, rewards)

    # ------------------------------------------------------------------
    # Training step (uses tensor buffer)
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
        """One gradient step. Samples from tensor replay buffer.

        Uses Double DQN with gamma > 0 for backward compatibility.
        TODO(open-question): With gamma=0 (correct for bandits), this simplifies
        to supervised regression: loss = huber(Q(s, a_selected), reward). The
        target network becomes unnecessary.
        """
        if len(self._buffer) < self.batch_size:
            return None

        states, actions, rewards = self._buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        # Current Q-values for selected actions
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max Q(s, a') — note next_state == state
        # TODO(open-question): With gamma=0, target = rewards (no target network).
        with torch.no_grad():
            next_actions = self.q_network(states).max(dim=1)[1]
            next_q = self.target_network(states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q

        loss = self.loss_fn(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        self.training_step += 1
        self.loss_history.append(loss.item())
        self.reward_history.append(rewards.mean().item())

        return loss.item()

    def update_target_network(self):
        """Update target network parameters."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_batch(self, states: torch.Tensor, labels: torch.Tensor) -> dict:
        """Vectorized validation on a batch of states.

        Unlike training (which uses alpha > 0.5 as prediction), validation
        computes the proper fused score: (1 - alpha) * anomaly + alpha * gat_prob.

        Args:
            states: [N, D] tensor
            labels: [N] tensor

        Returns:
            Dict with accuracy, avg_reward, avg_alpha, alpha_std.
        """
        was_training = self.q_network.training
        self.q_network.eval()

        actions, alphas, norm_states = self.select_action_batch(states, training=False)

        # Proper fused prediction (matches original validate_agent)
        anomaly_scores, gat_probs = self._derive_scores_batch(norm_states)
        fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
        preds = (fused_scores > 0.5).long()

        correct = (preds == labels).sum().item()
        rewards = self.compute_fusion_reward_batch(preds, labels, norm_states, alphas)

        if was_training:
            self.q_network.train()

        result = {
            "accuracy": correct / len(labels),
            "avg_reward": rewards.mean().item(),
            "avg_alpha": alphas.mean().item(),
            "alpha_std": alphas.std().item(),
        }

        # Update LR scheduler
        self.scheduler.step(result["avg_reward"])

        # Early stopping tracking
        current_score = result["accuracy"] + 0.1 * result["avg_reward"]
        if current_score > self.best_validation_score:
            self.best_validation_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        self.validation_scores.append(result)
        return result

    def validate_agent(self, validation_data: list[tuple], num_samples: int = 1000) -> dict:
        """Legacy single-sample validation. Prefer validate_batch for training."""
        self.q_network.eval()

        correct = 0
        total_reward = 0
        alpha_values_used = []

        sample_data = (
            validation_data[:num_samples]
            if len(validation_data) >= num_samples
            else validation_data
        )

        if not sample_data:
            self.q_network.train()
            return {"accuracy": 0.0, "avg_reward": 0.0, "avg_alpha": 0.0, "alpha_std": 0.0}

        for state_features, true_label in sample_data:
            alpha, _, _ = self.select_action(state_features, training=False)
            alpha_values_used.append(alpha)

            anomaly_score, gat_prob = self._derive_scores(state_features)
            fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
            prediction = 1 if fused_score > 0.5 else 0

            correct += prediction == true_label
            reward = self.compute_fusion_reward(prediction, true_label, state_features, alpha)
            total_reward += reward

        self.q_network.train()

        validation_results = {
            "accuracy": correct / len(sample_data),
            "avg_reward": total_reward / len(sample_data),
            "avg_alpha": np.mean(alpha_values_used),
            "alpha_std": np.std(alpha_values_used),
        }

        self.scheduler.step(validation_results["avg_reward"])

        current_score = validation_results["accuracy"] + 0.1 * validation_results["avg_reward"]
        if current_score > self.best_validation_score:
            self.best_validation_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        self.validation_scores.append(validation_results)
        return validation_results

    @property
    def buffer_size_current(self) -> int:
        """Current number of experiences in the replay buffer."""
        return len(self._buffer)


# ---------------------------------------------------------------------------
# MLP Fusion Agent
# ---------------------------------------------------------------------------


class MLPFusionNetwork(nn.Module):
    """Simple MLP for binary classification from fusion state vectors."""

    def __init__(self, state_dim: int, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPFusionAgent(FusionAgent):
    """Supervised MLP baseline: learns binary classification directly from state vectors.

    Same 15-D state as DQN, but trained with BCE loss instead of RL episodes.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        lr: float = 0.001,
        device: str = "cpu",
    ):
        self.device = device
        self.model = MLPFusionNetwork(state_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train_on_cache(self, train_states, train_labels, val_states, val_labels, cfg) -> float:
        max_epochs = cfg.fusion.mlp_max_epochs
        batch_size = cfg.dqn.batch_size
        best_acc = 0.0

        for epoch in range(max_epochs):
            self.model.train()
            idx = torch.randperm(len(train_states))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(train_states), batch_size):
                batch_idx = idx[start : start + batch_size]
                states = train_states[batch_idx].to(self.device)
                labels = train_labels[batch_idx].float().to(self.device)

                logits = self.model(states)
                loss = self.loss_fn(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            if (epoch + 1) % 10 == 0:
                acc = self._evaluate(val_states, val_labels)
                log.info(
                    "MLP epoch %d/%d  loss=%.4f  val_acc=%.4f",
                    epoch + 1,
                    max_epochs,
                    epoch_loss / max(n_batches, 1),
                    acc,
                )
                if acc > best_acc:
                    best_acc = acc

        return best_acc

    def _evaluate(self, states: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(states.to(self.device))
            preds = (logits > 0).long()
            correct = (preds == labels.to(self.device)).sum().item()
        return correct / len(labels)

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict()}

    def fuse(self, state_features: np.ndarray) -> int:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            logit = self.model(t)
            return 1 if logit.item() > 0 else 0


# ---------------------------------------------------------------------------
# Weighted Average Fusion Agent
# ---------------------------------------------------------------------------


class WeightedAvgFusionAgent(FusionAgent):
    """Simplest baseline: learns a single scalar alpha per model.

    If this matches DQN's F1, the RL approach is unjustified.
    Fusion: score = (1 - sigmoid(w)) * vgae_conf + sigmoid(w) * gat_conf
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.weight = nn.Parameter(torch.zeros(1, device=device))
        self.optimizer = optim.Adam([self.weight], lr=0.01)
        self.loss_fn = nn.BCELoss()

        from .registry import feature_layout

        layout = feature_layout()
        self._vgae_conf_idx = layout["vgae"][2]
        self._gat_conf_idx = layout["gat"][2]

    def train_on_cache(self, train_states, train_labels, val_states, val_labels, cfg) -> float:
        max_epochs = cfg.fusion.mlp_max_epochs
        best_acc = 0.0

        for epoch in range(max_epochs):
            alpha = torch.sigmoid(self.weight)
            vgae_conf = train_states[:, self._vgae_conf_idx].to(self.device)
            gat_conf = train_states[:, self._gat_conf_idx].to(self.device)
            scores = (1 - alpha) * vgae_conf + alpha * gat_conf
            scores = torch.clamp(scores, 1e-7, 1 - 1e-7)
            labels = train_labels.float().to(self.device)

            loss = self.loss_fn(scores, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                acc = self._evaluate(val_states, val_labels)
                a = torch.sigmoid(self.weight).item()
                log.info(
                    "WeightedAvg epoch %d/%d  loss=%.4f  val_acc=%.4f  alpha=%.3f",
                    epoch + 1,
                    max_epochs,
                    loss.item(),
                    acc,
                    a,
                )
                if acc > best_acc:
                    best_acc = acc

        return best_acc

    def _evaluate(self, states: torch.Tensor, labels: torch.Tensor) -> float:
        with torch.no_grad():
            alpha = torch.sigmoid(self.weight)
            vgae_conf = states[:, self._vgae_conf_idx].to(self.device)
            gat_conf = states[:, self._gat_conf_idx].to(self.device)
            scores = (1 - alpha) * vgae_conf + alpha * gat_conf
            preds = (scores > 0.5).long()
            correct = (preds == labels.to(self.device)).sum().item()
        return correct / len(labels)

    def state_dict(self) -> dict:
        return {"weight": self.weight.detach().cpu(), "alpha": torch.sigmoid(self.weight).item()}

    def fuse(self, state_features: np.ndarray) -> int:
        with torch.no_grad():
            alpha = torch.sigmoid(self.weight).item()
            vgae_conf = state_features[self._vgae_conf_idx]
            gat_conf = state_features[self._gat_conf_idx]
            score = (1 - alpha) * vgae_conf + alpha * gat_conf
            return 1 if score > 0.5 else 0
