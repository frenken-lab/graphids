from __future__ import annotations

import structlog
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

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
        weight_decay=1e-5,
        scheduler_patience=1000,
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
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=scheduler_patience, factor=0.8
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability

        # Tensor replay buffer (replaces deque of tuples)
        self._buffer = TensorReplayBuffer(buffer_size, state_dim)

        # Training tracking
        self.training_step = 0
        self.update_counter = 0

        # Derive feature indices from registry (no hardcoded offsets)
        from .registry import feature_layout

        layout = feature_layout()
        vgae = layout["vgae"]
        gat = layout["gat"]
        self._confidence_indices = [fl.confidence_idx for fl in layout.values()]
        self._vgae_error_slice = slice(vgae.offset, vgae.offset + 3)
        self._gat_logit_slice = slice(gat.offset, gat.offset + 2)
        self._vgae_conf_idx = vgae.confidence_idx
        self._gat_conf_idx = gat.confidence_idx

        # Weights for VGAE anomaly score (used in batch reward computation)
        self._vgae_weights = None  # Set via from_config() or set_vgae_weights()

        log.info("dqn_agent_initialized", actions=alpha_steps, state_dim=self.state_dim)

    def set_vgae_weights(self, weights: tuple[float, ...]) -> None:
        """Set VGAE error weights for anomaly score derivation."""
        self._vgae_weights = torch.tensor(weights, dtype=torch.float32)

    @classmethod
    def from_config(
        cls,
        cfg,
        device: str = "cpu",
        *,
        inference: bool = False,
    ) -> "EnhancedDQNFusionAgent":
        """Create agent from config. Set inference=True for eval/serve (no exploration)."""
        from .registry import fusion_state_dim

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
        )
        if inference:
            kwargs.update(epsilon=0.0, epsilon_decay=1.0, min_epsilon=0.0)
        else:
            kwargs.update(
                epsilon=cfg.dqn.epsilon,
                epsilon_decay=cfg.dqn.epsilon_decay,
                min_epsilon=cfg.dqn.min_epsilon,
            )
        agent = cls(**kwargs)
        agent.set_vgae_weights(cfg.dqn.vgae_error_weights)
        return agent

    # ------------------------------------------------------------------
    # Single-sample methods (inference / serve.py)
    # ------------------------------------------------------------------

    def normalize_state(self, state_features: np.ndarray) -> np.ndarray:
        """Normalize a single state (numpy). Delegates to batch path."""
        if not isinstance(state_features, np.ndarray):
            state_features = np.array(state_features, dtype=np.float32)
        if len(state_features) != self.state_dim:
            raise ValueError(f"Expected {self.state_dim}D state, got {len(state_features)}D")
        t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
        return self._normalize_batch(t).squeeze(0).numpy()

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
        """Derive anomaly_score and gat_prob from a single state (numpy). Delegates to batch path."""
        t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
        anomaly, gat_prob = self._derive_scores_batch(t)
        return float(anomaly[0]), float(gat_prob[0])

    def compute_fusion_reward(
        self, prediction: int, true_label: int, state_features: np.ndarray, alpha: float
    ) -> float:
        """Compute reward for a single sample. Delegates to batch method."""
        t_state = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
        t_pred = torch.tensor([prediction], dtype=torch.long)
        t_label = torch.tensor([true_label], dtype=torch.long)
        t_alpha = torch.tensor([alpha], dtype=torch.float32)
        reward = self.compute_fusion_reward_batch(t_pred, t_label, t_state, t_alpha)
        return float(reward[0])

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
        if self._vgae_weights is None:
            self._vgae_weights = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float32)
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

        return result

    def load_checkpoint(self, checkpoint_or_path: dict | str | Path) -> None:
        """Load Q-network and target network weights.

        Accepts either a pre-loaded state_dict (from ArtifactMapper) or a
        filesystem path (legacy convenience).
        """
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


class MLPFusionModule(pl.LightningModule):
    """Supervised MLP baseline: binary classification from fusion state vectors.

    Same state as DQN, but trained with BCE loss via Lightning instead of RL episodes.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        lr: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPFusionNetwork(state_dim, hidden_dims)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, labels = batch
        logits = self(states)
        loss = self.loss_fn(logits, labels.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        logits = self(states)
        loss = self.loss_fn(logits, labels.float())
        preds = (logits > 0).long()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def fuse(self, state_features: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            logit = self(t)
            return 1 if logit.item() > 0 else 0


# ---------------------------------------------------------------------------
# Weighted Average Fusion Agent
# ---------------------------------------------------------------------------


class WeightedAvgModule(pl.LightningModule):
    """Simplest baseline: learns a single scalar alpha per model.

    If this matches DQN's F1, the RL approach is unjustified.
    Fusion: score = (1 - sigmoid(w)) * vgae_conf + sigmoid(w) * gat_conf
    """

    def __init__(self, lr: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.weight = nn.Parameter(torch.zeros(1))
        self.loss_fn = nn.BCELoss()
        self.lr = lr

        from .registry import feature_layout

        layout = feature_layout()
        self._vgae_conf_idx = layout["vgae"].confidence_idx
        self._gat_conf_idx = layout["gat"].confidence_idx

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.weight)
        vgae_conf = states[:, self._vgae_conf_idx]
        gat_conf = states[:, self._gat_conf_idx]
        return torch.clamp((1 - alpha) * vgae_conf + alpha * gat_conf, 1e-7, 1 - 1e-7)

    def training_step(self, batch, batch_idx):
        states, labels = batch
        scores = self(states)
        loss = self.loss_fn(scores, labels.float())
        self.log("train_loss", loss, prog_bar=True)
        self.log("alpha", torch.sigmoid(self.weight).item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        scores = self(states)
        loss = self.loss_fn(scores, labels.float())
        preds = (scores > 0.5).long()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def state_dict_for_save(self) -> dict:
        return {"weight": self.weight.detach().cpu(), "alpha": torch.sigmoid(self.weight).item()}

    def fuse(self, state_features: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            alpha = torch.sigmoid(self.weight).item()
            vgae_conf = state_features[self._vgae_conf_idx]
            gat_conf = state_features[self._gat_conf_idx]
            score = (1 - alpha) * vgae_conf + alpha * gat_conf
            return 1 if score > 0.5 else 0
