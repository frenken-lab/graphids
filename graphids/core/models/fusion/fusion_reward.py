"""Fusion reward calculation + shared fused prediction helper.

Reward shaping constants are fixed methodological choices — see
``kd-gat-paper/content/methodology.md §Stage 3 Adaptive Fusion``.
DQN and bandit share this identical reward; it is not an ablation axis.
"""

from __future__ import annotations

import torch

# --- Reward shaping constants (paper methodology.md eq-reward) ---------------
# Base rewards appear inline in the paper's reward equation (±3.0).
# The remaining coefficients weight the symbolic bonus/penalty terms
# (r_agree, r_conf, r_disagree, r_overconf + implicit balance bonus).
_REWARD_CORRECT = 3.0
_REWARD_INCORRECT = -3.0
_CONFIDENCE_WEIGHT = 0.5
_COMBINED_CONF_WEIGHT = 0.3
_DISAGREEMENT_PENALTY = -1.0
_OVERCONF_PENALTY = -1.5
_BALANCE_WEIGHT = 0.3


class FusionRewardCalculator(torch.nn.Module):
    """Vectorized fusion reward from state features, predictions, and labels.

    Extends nn.Module so ``_vgae_weights`` auto-transfers to GPU when the owning
    LightningModule (BanditFusionModule, DQNFusionModule) is moved to a device.

    The only tunable parameter is ``vgae_weights`` — the convex combination
    weights for the three VGAE reconstruction error components (recon, nbr,
    canid). All other shaping coefficients are fixed by the paper.
    """

    def __init__(
        self,
        *,
        vgae_weights: list[float] | tuple[float, ...],
    ) -> None:
        super().__init__()
        from .fusion_features import LAYOUT

        vgae = LAYOUT["vgae"]
        gat = LAYOUT["gat"]
        self._confidence_indices = [fl.confidence_idx for fl in LAYOUT.values()]
        self._vgae_error_slice = slice(vgae.offset, vgae.offset + 3)
        self._gat_logit_slice = slice(gat.offset, gat.offset + 2)
        self._vgae_conf_idx = vgae.confidence_idx
        self._gat_conf_idx = gat.confidence_idx

        self.register_buffer(
            "_vgae_weights", torch.tensor(vgae_weights, dtype=torch.float32)
        )

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
        """Vectorized reward computation.

        All inputs [N] or [N, D]. Returns [N].
        """
        anomaly_scores, gat_probs = self.derive_scores(states)
        vgae_conf = states[:, self._vgae_conf_idx]
        gat_conf = states[:, self._gat_conf_idx]
        combined_conf = torch.max(vgae_conf, gat_conf)

        correct = preds == labels
        base_reward = torch.where(correct, _REWARD_CORRECT, _REWARD_INCORRECT)
        model_agreement = 1.0 - (anomaly_scores - gat_probs).abs()

        # Correct path
        max_score = torch.max(anomaly_scores, gat_probs)
        confidence = torch.where(labels == 1, max_score, 1.0 - max_score)
        confidence_bonus = (
            _CONFIDENCE_WEIGHT * confidence + _COMBINED_CONF_WEIGHT * combined_conf
        )
        correct_reward = base_reward + model_agreement + confidence_bonus

        # Wrong path
        disagreement_term = _DISAGREEMENT_PENALTY * (1.0 - model_agreement)
        fused_confidence = alphas * gat_probs + (1 - alphas) * anomaly_scores
        overconf_term = torch.where(
            preds == 1,
            _OVERCONF_PENALTY * fused_confidence,
            _OVERCONF_PENALTY * (1.0 - fused_confidence),
        )
        wrong_reward = base_reward + disagreement_term + overconf_term

        total_reward = torch.where(correct, correct_reward, wrong_reward)
        balance_bonus = _BALANCE_WEIGHT * (1.0 - (alphas - 0.5).abs() * 2)
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
