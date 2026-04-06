"""Fusion reward calculation + shared fused prediction helper.

Reward shaping constants are fixed methodological choices — see
``kd-gat-paper/content/methodology.md §Stage 3 Adaptive Fusion``.
DQN and bandit share this identical reward; it is not an ablation axis.

The seven shaping coefficients + ``vgae_weights`` are sourced from
``configs/fusion/reward.libsonnet`` (imported by the method libsonnets)
so they land in ``run_record.json`` for reproducibility. If the constants
are omitted, they are loaded from the libsonnet (no hardcoded defaults).
"""

from __future__ import annotations

from functools import lru_cache

import torch


class FusionRewardCalculator(torch.nn.Module):
    """Vectorized fusion reward from state features, predictions, and labels.

    Extends nn.Module so ``_vgae_weights`` auto-transfers to GPU when the owning
    LightningModule (BanditFusionModule, DQNFusionModule) is moved to a device.

    All shaping coefficients are required kwargs — callers pass them from
    ``configs/fusion/reward.libsonnet`` via ``reward_kwargs``.
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
        from .fusion_features import LAYOUT

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


_REWARD_DEFAULT_KEYS = (
    "correct",
    "incorrect",
    "confidence_weight",
    "combined_conf_weight",
    "disagreement_penalty",
    "overconf_penalty",
    "balance_weight",
)


@lru_cache(maxsize=1)
def _reward_defaults() -> dict[str, float]:
    from graphids.config.constants import PROJECT_ROOT
    from graphids.config.jsonnet import render

    reward_path = PROJECT_ROOT / "configs" / "fusion" / "reward.libsonnet"
    return render(reward_path)


def resolve_reward_kwargs(reward_kwargs: dict | None) -> dict:
    kwargs = dict(reward_kwargs or {})
    defaults = _reward_defaults()
    for key in _REWARD_DEFAULT_KEYS:
        if key not in kwargs:
            kwargs[key] = defaults[key]
    return kwargs


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
