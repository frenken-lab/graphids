"""Shared reward computation for fusion agents (DQN, bandit, etc).

Extracted from EnhancedDQNFusionAgent to allow reuse across fusion methods.
"""

from __future__ import annotations

import structlog
import torch

log = structlog.get_logger()


class FusionRewardCalculator:
    """Vectorized fusion reward from state features, predictions, and labels.

    Encapsulates the feature layout indices and VGAE error weights so callers
    don't need to thread them through every call.

    Args:
        vgae_weights: Weights for combining VGAE reconstruction errors into a
            single anomaly score. Must sum to ~1.0. Required -- callers must
            provide explicitly (typically from cfg.dqn.vgae_error_weights).
        reward_correct: Base reward for correct predictions.
        reward_incorrect: Base reward for incorrect predictions.
        confidence_weight: Weight on per-model confidence in correct-path bonus.
        combined_conf_weight: Weight on max(vgae_conf, gat_conf) in correct-path bonus.
        disagreement_penalty: Multiplier on model disagreement in wrong-path penalty.
        overconf_penalty: Multiplier on overconfidence in wrong-path penalty.
        balance_weight: Weight on alpha-balance bonus (penalizes extreme alphas).
    """

    def __init__(
        self,
        *,
        vgae_weights: list[float] | tuple[float, ...],
        reward_correct: float = 3.0,
        reward_incorrect: float = -3.0,
        confidence_weight: float = 0.5,
        combined_conf_weight: float = 0.3,
        disagreement_penalty: float = -1.0,
        overconf_penalty: float = -1.5,
        balance_weight: float = 0.3,
    ) -> None:
        from .registry import feature_layout

        layout = feature_layout()
        vgae = layout["vgae"]
        gat = layout["gat"]
        self._confidence_indices = [fl.confidence_idx for fl in layout.values()]
        self._vgae_error_slice = slice(vgae.offset, vgae.offset + 3)
        self._gat_logit_slice = slice(gat.offset, gat.offset + 2)
        self._vgae_conf_idx = vgae.confidence_idx
        self._gat_conf_idx = gat.confidence_idx

        self._vgae_weights = torch.tensor(vgae_weights, dtype=torch.float32)

        # Reward shaping coefficients
        self._reward_correct = reward_correct
        self._reward_incorrect = reward_incorrect
        self._confidence_weight = confidence_weight
        self._combined_conf_weight = combined_conf_weight
        self._disagreement_penalty = disagreement_penalty
        self._overconf_penalty = overconf_penalty
        self._balance_weight = balance_weight

    def set_vgae_weights(self, weights: tuple[float, ...] | list[float]) -> None:
        """Set VGAE error weights for anomaly score derivation."""
        self._vgae_weights = torch.tensor(weights, dtype=torch.float32)

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        """Clamp confidence features to [0, 1]. Returns a new tensor."""
        states = states.clone().float()
        for idx in self._confidence_indices:
            states[:, idx].clamp_(0.0, 1.0)
        return states

    def derive_scores(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive anomaly_score and gat_prob from state features.

        [N, D] -> ([N], [N]).
        """
        vgae_errors = states[:, self._vgae_error_slice]
        anomaly_scores = (
            (vgae_errors * self._vgae_weights).sum(dim=1).clamp(0.0, 1.0)
        )

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
        """Vectorized reward computation.

        All inputs [N] or [N, D]. Returns [N].
        """
        anomaly_scores, gat_probs = self.derive_scores(states)
        vgae_conf = states[:, self._vgae_conf_idx]
        gat_conf = states[:, self._gat_conf_idx]
        combined_conf = torch.max(vgae_conf, gat_conf)

        correct = preds == labels
        base_reward = torch.where(
            correct, self._reward_correct, self._reward_incorrect
        )
        model_agreement = 1.0 - (anomaly_scores - gat_probs).abs()

        # Correct path
        max_score = torch.max(anomaly_scores, gat_probs)
        confidence = torch.where(labels == 1, max_score, 1.0 - max_score)
        confidence_bonus = (
            self._confidence_weight * confidence
            + self._combined_conf_weight * combined_conf
        )
        correct_reward = base_reward + model_agreement + confidence_bonus

        # Wrong path
        disagreement_term = self._disagreement_penalty * (
            1.0 - model_agreement
        )
        fused_confidence = alphas * gat_probs + (1 - alphas) * anomaly_scores
        overconf_term = torch.where(
            preds == 1,
            self._overconf_penalty * fused_confidence,
            self._overconf_penalty * (1.0 - fused_confidence),
        )
        wrong_reward = base_reward + disagreement_term + overconf_term

        total_reward = torch.where(correct, correct_reward, wrong_reward)
        balance_bonus = self._balance_weight * (
            1.0 - (alphas - 0.5).abs() * 2
        )
        return total_reward + balance_bonus


def fused_predict(agent, states: torch.Tensor) -> dict:
    """Greedy fused prediction shared by DQN and bandit agents.

    Requires agent to have: select_action_batch, reward_calc, decision_threshold.
    """
    actions, alphas, norm_states = agent.select_action_batch(states, training=False)
    anomaly_scores, gat_probs = agent.reward_calc.derive_scores(norm_states)
    fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
    preds = (fused_scores > agent.decision_threshold).long()
    return {"preds": preds, "fused_scores": fused_scores, "alphas": alphas, "norm_states": norm_states}
