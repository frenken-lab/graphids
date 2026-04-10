"""Fusion model family base — shared RL training loop, reward, and NN utilities.

FusionModuleBase: RL subclasses implement ``select_action_batch`` and
``train_episode``. Everything else — training_step, predict, validate_batch,
test hooks, reward computation, replay buffer — lives here.

Supervised subclasses (MLP, WeightedAvg) override training_step /
validation_step / test_step with standard loss-based flows.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from graphids.core.trainer import MetricAccumulator

from ..base import LAYOUT, STATE_DIM, binary_test_metrics

# ---------------------------------------------------------------------------
# NN building blocks (shared by bandit + DQN)
# ---------------------------------------------------------------------------


def build_mlp_body(state_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    """Build MLP trunk: [Linear -> LayerNorm -> ReLU -> Dropout(0.2)] x N."""
    layers: list[nn.Module] = []
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
    return nn.Sequential(*layers)


class TensorReplayBuffer:
    """Fixed-size circular buffer backed by contiguous tensors.

    Stores (state, action, reward) triples only — next_state is always
    identical to state in the current fusion formulation.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.states = torch.zeros(capacity, state_dim)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self._pos = 0
        self._size = 0

    def add_batch(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        n = len(states)
        if n >= self.capacity:
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
        idx = torch.randint(0, self._size, (batch_size,))
        return self.states[idx], self.actions[idx], self.rewards[idx]

    def __len__(self):
        return self._size


# ---------------------------------------------------------------------------
# Reward calculator (shared by bandit + DQN)
# ---------------------------------------------------------------------------


class FusionRewardCalculator(torch.nn.Module):
    """Vectorized fusion reward from state features, predictions, and labels.

    Extends nn.Module so ``_vgae_weights`` auto-transfers to GPU when the owning
    module is moved to a device.

    All shaping coefficients are required kwargs — sourced from
    ``configs/models/fusion/reward.libsonnet`` via the jsonnet config.
    """

    def __init__(
        self,
        *,
        vgae_weights: list[float] | tuple[float, ...],
        correct: float,
        incorrect: float,
        confidence_weight: float,
        combined_conf_weight: float,
        disagreement_penalty: float,
        overconf_penalty: float,
        balance_weight: float,
    ) -> None:
        super().__init__()

        vgae = LAYOUT["vgae"]
        gat = LAYOUT["gat"]
        self._confidence_indices = [fl.confidence_idx for fl in LAYOUT.values()]
        self._vgae_error_slice = slice(vgae.offset, vgae.offset + 3)
        self._gat_logit_slice = slice(gat.offset, gat.offset + 2)
        self._vgae_conf_idx = vgae.confidence_idx
        self._gat_conf_idx = gat.confidence_idx

        self.register_buffer("_vgae_weights", torch.tensor(vgae_weights, dtype=torch.float32))

        self._reward_correct = correct
        self._reward_incorrect = incorrect
        self._confidence_weight = confidence_weight
        self._combined_conf_weight = combined_conf_weight
        self._disagreement_penalty = disagreement_penalty
        self._overconf_penalty = overconf_penalty
        self._balance_weight = balance_weight

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        """Clamp confidence features to [0, 1]. Returns a new tensor."""
        states = states.clone().float()
        for idx in self._confidence_indices:
            states[:, idx].clamp_(0.0, 1.0)
        return states

    def derive_scores(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive anomaly_score and gat_prob from state features. [N, D] -> ([N], [N])."""
        vgae_errors = states[:, self._vgae_error_slice]
        anomaly_scores = (vgae_errors * self._vgae_weights).sum(dim=1).clamp(0.0, 1.0)

        gat_logits = states[:, self._gat_logit_slice]
        gat_probs = torch.softmax(gat_logits, dim=1)[:, 1]

        return anomaly_scores, gat_probs

    def compute(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        states: torch.Tensor,
        alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized reward computation. All inputs [N] or [N, D]. Returns [N]."""
        anomaly_scores, gat_probs = self.derive_scores(states)
        vgae_conf = states[:, self._vgae_conf_idx]
        gat_conf = states[:, self._gat_conf_idx]
        combined_conf = torch.max(vgae_conf, gat_conf)

        correct = preds == labels
        base_reward = torch.where(correct, self._reward_correct, self._reward_incorrect)
        model_agreement = 1.0 - (anomaly_scores - gat_probs).abs()

        # Correct path
        max_score = torch.max(anomaly_scores, gat_probs)
        confidence = torch.where(labels == 1, max_score, 1.0 - max_score)
        confidence_bonus = (
            self._confidence_weight * confidence + self._combined_conf_weight * combined_conf
        )
        correct_reward = base_reward + model_agreement + confidence_bonus

        # Wrong path
        disagreement_term = self._disagreement_penalty * (1.0 - model_agreement)
        fused_confidence = alphas * gat_probs + (1 - alphas) * anomaly_scores
        overconf_term = torch.where(
            preds == 1,
            self._overconf_penalty * fused_confidence,
            self._overconf_penalty * (1.0 - fused_confidence),
        )
        wrong_reward = base_reward + disagreement_term + overconf_term

        total_reward = torch.where(correct, correct_reward, wrong_reward)
        balance_bonus = self._balance_weight * (1.0 - (alphas - 0.5).abs() * 2)
        return total_reward + balance_bonus


def fused_predict(agent, states: torch.Tensor) -> dict:
    """Greedy fused prediction shared by DQN and bandit agents.

    Requires agent to have: select_action_batch, reward_calc, decision_threshold.
    """
    actions, alphas, norm_states = agent.select_action_batch(states, training=False)
    anomaly_scores, gat_probs = agent.reward_calc.derive_scores(norm_states)
    fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
    preds = (fused_scores > agent.decision_threshold).long()
    return {
        "preds": preds,
        "fused_scores": fused_scores,
        "alphas": alphas,
        "norm_states": norm_states,
    }


# ---------------------------------------------------------------------------
# Fusion module base class
# ---------------------------------------------------------------------------


class FusionModuleBase(nn.Module):
    """Base for all fusion models. RL subclasses get the full training loop
    for free by implementing ``select_action_batch`` and ``train_episode``."""

    # RL subclasses (Bandit, DQN) set this to False
    automatic_optimization = False

    @staticmethod
    def _capture_hparams(local_vars: dict[str, Any], ignore: tuple[str, ...] = ()) -> SimpleNamespace:
        """Capture ``__init__`` kwargs as a ``SimpleNamespace``."""
        skip = {"self", "__class__", *ignore}
        return SimpleNamespace(**{k: v for k, v in local_vars.items() if k not in skip})

    def __init__(
        self,
        *,
        state_dim: int = STATE_DIM,
        alpha_steps: int = 21,
        batch_size: int = 128,
        buffer_size: int = 100_000,
        decision_threshold: float = 0.5,
        reward_kwargs: dict | None = None,
    ):
        super().__init__()
        self._metric_acc = MetricAccumulator()
        self._trainer = None
        # Non-persistent buffer that tracks device through .to()/.cuda()/.cpu()
        self.register_buffer("_device_tracker", torch.empty(0), persistent=False)
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.decision_threshold = decision_threshold

        self.register_buffer("alpha_values", torch.linspace(0, 1, alpha_steps))
        self.alpha_steps = alpha_steps

        if reward_kwargs is not None:
            self.reward_calc = FusionRewardCalculator(**reward_kwargs)
        self._buffer = TensorReplayBuffer(buffer_size, state_dim)

        self.test_metrics = binary_test_metrics()

    @property
    def device(self) -> torch.device:
        return self._device_tracker.device

    # -- logging (replaces pl self.log / self.log_dict) ----------------------

    def log(self, name: str, value: Any, *, batch_size: int = 1, **_kwargs) -> None:
        v = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
        self._metric_acc.update(name, v, batch_size)

    def log_dict(self, metrics: dict[str, Any], **kwargs) -> None:
        for k, v in metrics.items():
            self.log(k, v, **kwargs)

    # -- setup (no-op by default, overridden if needed) ----------------------

    def setup(self, datamodule=None):
        pass

    def build_optimizers(self, max_epochs: int) -> tuple[torch.optim.Optimizer | None, Any]:
        """Default: no optimizer (RL models manage their own). Override in subclasses."""
        return None, None

    # -- Subclass contract ---------------------------------------------------

    def select_action_batch(
        self, states: torch.Tensor, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick actions for a batch of states. Returns (actions, alphas, norm_states)."""
        raise NotImplementedError

    def train_episode(self, states: torch.Tensor, labels: torch.Tensor) -> dict:
        """One training episode. Returns metric dict for logging."""
        raise NotImplementedError

    # -- Training hooks (shared by all RL fusion) ----------------------------

    def training_step(self, batch, batch_idx):
        states, labels = batch
        result = self.train_episode(states, labels)
        for k, v in result.items():
            if v is not None:
                self.log(k, float(v), prog_bar=(k in ("avg_reward", "accuracy")))

    def predict(self, states: torch.Tensor) -> dict:
        """Greedy fused prediction (no exploration)."""
        return fused_predict(self, states)

    def validate_batch(self, states: torch.Tensor, labels: torch.Tensor) -> dict:
        """Greedy evaluation — shared by bandit and DQN."""
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

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        metrics = self.validate_batch(states, labels)
        self.log("val_acc", metrics.get("accuracy", 0.0), prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        states, labels = batch
        result = self.predict(states)
        self.test_metrics.update(result["preds"], labels)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def on_save_checkpoint(self, checkpoint):
        pass

    def on_load_checkpoint(self, checkpoint):
        pass
