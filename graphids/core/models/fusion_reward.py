"""Shared reward computation for fusion agents (DQN, bandit, etc).

Extracted from EnhancedDQNFusionAgent to allow reuse across fusion methods.
"""

from __future__ import annotations

import torch


class FusionRewardCalculator:
    """Vectorized fusion reward from state features, predictions, and labels.

    Encapsulates the feature layout indices and VGAE error weights so callers
    don't need to thread them through every call.
    """

    def __init__(self) -> None:
        from .registry import feature_layout

        layout = feature_layout()
        vgae = layout["vgae"]
        gat = layout["gat"]
        self._confidence_indices = [fl.confidence_idx for fl in layout.values()]
        self._vgae_error_slice = slice(vgae.offset, vgae.offset + 3)
        self._gat_logit_slice = slice(gat.offset, gat.offset + 2)
        self._vgae_conf_idx = vgae.confidence_idx
        self._gat_conf_idx = gat.confidence_idx
        self._vgae_weights: torch.Tensor | None = None

    def set_vgae_weights(self, weights: tuple[float, ...] | list[float]) -> None:
        """Set VGAE error weights for anomaly score derivation."""
        self._vgae_weights = torch.tensor(weights, dtype=torch.float32)

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        """Clamp confidence features to [0, 1]. Returns a new tensor."""
        states = states.clone().float()
        for idx in self._confidence_indices:
            states[:, idx].clamp_(0.0, 1.0)
        return states

    def derive_scores(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive anomaly_score and gat_prob from state features. [N, D] → ([N], [N])."""
        if self._vgae_weights is None:
            self._vgae_weights = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float32)
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
        base_reward = torch.where(correct, 3.0, -3.0)
        model_agreement = 1.0 - (anomaly_scores - gat_probs).abs()

        # Correct path
        max_score = torch.max(anomaly_scores, gat_probs)
        confidence = torch.where(labels == 1, max_score, 1.0 - max_score)
        confidence_bonus = 0.5 * confidence + 0.3 * combined_conf
        correct_reward = base_reward + model_agreement + confidence_bonus

        # Wrong path
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
