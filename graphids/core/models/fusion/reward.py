"""Fusion reward calculator. Reads named keys from the feature TensorDict.

Two reward variants share the same ``compute()`` API so RLFusionBase doesn't
care which is in use:

- ``FusionRewardCalculator`` — legacy 2-way (correct/incorrect) base reward
  with agreement / confidence / disagreement / overconfidence / balance
  shaping terms. Pre-2026-05-07.

- ``MinimalFusionRewardCalculator`` — PBRS-compliant 4-way (TP/TN/FP/FN)
  asymmetric base reward + attack-gated confidence bonus. No shaping terms
  that violate Ng-Harada-Russell 1999 policy invariance. Phase 1.1 of the
  fusion improvement plan.

``FusionRewardCalculator.from_kwargs(**reward_kwargs)`` is the dispatch
entry point — picks calculator class on the optional ``mode`` field
(default ``"legacy"`` for backward compat).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


class FusionRewardCalculator(torch.nn.Module):
    """Vectorized fusion reward over a feature TensorDict.

    Required nested keys:
      - ``("vgae", "errors")`` [N, 3] — recon, mahal, kl
      - ``("vgae", "conf")``   [N, 1]
      - ``("gat", "probs")``   [N, 2]
      - ``("gat", "conf")``    [N, 1]

    Other keys (z_stats, emb_stats, …) are ignored — they're consumed by
    the supervised/Q-network paths after flattening, not by the reward.
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
        self.register_buffer("_vgae_weights", torch.tensor(vgae_weights, dtype=torch.float32))
        self._reward_correct = correct
        self._reward_incorrect = incorrect
        self._confidence_weight = confidence_weight
        self._combined_conf_weight = combined_conf_weight
        self._disagreement_penalty = disagreement_penalty
        self._overconf_penalty = overconf_penalty
        self._balance_weight = balance_weight

    @staticmethod
    def from_kwargs(**reward_kwargs) -> FusionRewardCalculator:
        """Factory: dispatch on ``mode`` (default ``"legacy"``).

        ``mode`` is consumed here, not forwarded — calculator subclasses
        don't accept it as a constructor arg. Existing rendered plans with
        no ``mode`` field continue to receive ``FusionRewardCalculator``.
        """
        mode = reward_kwargs.pop("mode", "legacy")
        if mode == "legacy":
            return FusionRewardCalculator(**reward_kwargs)
        if mode == "minimal":
            return MinimalFusionRewardCalculator(**reward_kwargs)
        raise ValueError(f"unknown reward mode: {mode!r} (expected 'legacy' or 'minimal')")

    def normalize(self, td: TensorDict) -> TensorDict:
        """Clamp confidence keys to [0, 1]. Returns a shallow-cloned TD."""
        out = td.clone(recurse=False)
        out["vgae"] = td["vgae"].clone(recurse=False)
        out["gat"] = td["gat"].clone(recurse=False)
        out["vgae", "conf"] = td["vgae", "conf"].clamp(0.0, 1.0)
        out["gat", "conf"] = td["gat", "conf"].clamp(0.0, 1.0)
        return out

    def derive_scores(self, td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (anomaly_scores[N], gat_probs_pos[N]).

        ``anomaly`` was previously ``(errors @ weights).clamp(0, 1)`` —
        saturated to 1.0 for nearly every sample because typical weighted
        error magnitudes are O(1)–O(10), well above the clamp ceiling. That
        broke the RL fusion path: at α≈0.5 the blended score
        ``α·gat_prob + (1−α)·anomaly`` was ≥0.5 everywhere → predict-attack
        on every sample → MCC≈0 even though AUROC was perfect (the ranking
        was right but the threshold was uniformly above benigns). Replace
        the clamp with the Möbius transform ``x/(1+x)``: bounded [0, 1) on
        non-negative errors (recon, mahal, kl are all non-negative), strictly
        monotonic, parameter-free, preserves rank ordering. This matches the
        sigmoidal compression ``weighted_avg`` already uses on `recon_mean`.
        """
        weighted = (td["vgae", "errors"] * self._vgae_weights).sum(dim=1)
        anomaly = weighted / (1.0 + weighted)
        gat_prob = td["gat", "probs"][:, 1]
        return anomaly, gat_prob

    def compute(
        self,
        td: TensorDict,
        preds: torch.Tensor,
        labels: torch.Tensor,
        alphas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Vectorized reward. Returns ``(total[N], components)``.

        ``components`` is a per-term breakdown of what each shaping term
        contributed to ``total`` per graph (mutually-exclusive correct/wrong
        terms zero out on the inactive branch). ``sum(components.values()) ==
        total`` by construction. Used by callers to log per-component means
        per epoch — diagnostic for which term the policy is exploiting.
        """
        anomaly, gat_prob = self.derive_scores(td)
        vgae_conf = td["vgae", "conf"].squeeze(-1)
        gat_conf = td["gat", "conf"].squeeze(-1)
        combined_conf = torch.max(vgae_conf, gat_conf)

        correct = preds == labels
        base = torch.where(correct, self._reward_correct, self._reward_incorrect)
        agreement = 1.0 - (anomaly - gat_prob).abs()

        max_score = torch.max(anomaly, gat_prob)
        confidence = torch.where(labels == 1, max_score, 1.0 - max_score)
        bonus = self._confidence_weight * confidence + self._combined_conf_weight * combined_conf

        disagreement = self._disagreement_penalty * (1.0 - agreement)
        fused = alphas * gat_prob + (1 - alphas) * anomaly
        overconf = torch.where(
            preds == 1,
            self._overconf_penalty * fused,
            self._overconf_penalty * (1.0 - fused),
        )
        balance = self._balance_weight * (1.0 - (alphas - 0.5).abs() * 2)

        zero = torch.zeros_like(base)
        components = {
            "r_classification": base,
            "r_agreement": torch.where(correct, agreement, zero),
            "r_confidence": torch.where(correct, bonus, zero),
            "r_disagreement_penalty": torch.where(correct, zero, disagreement),
            "r_overconfidence_penalty": torch.where(correct, zero, overconf),
            "r_balance": balance,
        }
        total = sum(components.values())
        return total, components


class MinimalFusionRewardCalculator(torch.nn.Module):
    """PBRS-compliant 4-way (TP/TN/FP/FN) reward + attack-gated confidence bonus.

    No ``balance`` / ``agreement`` / ``disagreement_penalty`` / ``combined_conf``
    shaping — those violate Ng-Harada-Russell 1999 policy invariance and were
    diagnosed (2026-05-06 fusion analysis) as the cause of the all-benign
    equilibrium under 86% benign and the constant-arm-20 RL collapse.

    Asymmetric FN/FP costs encode IDS deployment context: missed attacks
    (FN) cost ~4× false alarms (FP) per F2-optimization (Davis & Goadrich
    2006). Confidence bonus gated to attack predictions only — no benign
    inflation that would re-create the majority-class equilibrium.

    Same ``compute()`` signature as ``FusionRewardCalculator`` so the
    caller is reward-class-agnostic. Components dict uses the same key set
    as the legacy calculator (zero-fills the inactive shaping terms) for
    uniform MLflow logging across reward variants.
    """

    def __init__(
        self,
        *,
        vgae_weights: list[float] | tuple[float, ...],
        tp_reward: float = 3.0,
        tn_reward: float = 1.5,
        fp_cost: float = -1.5,
        fn_cost: float = -6.0,
        confidence_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.register_buffer("_vgae_weights", torch.tensor(vgae_weights, dtype=torch.float32))
        self._tp_reward = tp_reward
        self._tn_reward = tn_reward
        self._fp_cost = fp_cost
        self._fn_cost = fn_cost
        self._confidence_weight = confidence_weight

    def normalize(self, td: TensorDict) -> TensorDict:
        out = td.clone(recurse=False)
        out["vgae"] = td["vgae"].clone(recurse=False)
        out["gat"] = td["gat"].clone(recurse=False)
        out["vgae", "conf"] = td["vgae", "conf"].clamp(0.0, 1.0)
        out["gat", "conf"] = td["gat", "conf"].clamp(0.0, 1.0)
        return out

    def derive_scores(self, td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        # Same Möbius transform as legacy calculator — see its derive_scores
        # docstring for the saturation bug being fixed.
        weighted = (td["vgae", "errors"] * self._vgae_weights).sum(dim=1)
        anomaly = weighted / (1.0 + weighted)
        gat_prob = td["gat", "probs"][:, 1]
        return anomaly, gat_prob

    def compute(
        self,
        td: TensorDict,
        preds: torch.Tensor,
        labels: torch.Tensor,
        alphas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, gat_prob = self.derive_scores(td)
        attack_lbl = labels == 1
        attack_pred = preds == 1
        tp = attack_lbl & attack_pred
        tn = (~attack_lbl) & (~attack_pred)
        fp = (~attack_lbl) & attack_pred
        fn = attack_lbl & (~attack_pred)

        zero = torch.zeros_like(gat_prob)
        base = (
            torch.where(tp, torch.full_like(zero, self._tp_reward), zero)
            + torch.where(tn, torch.full_like(zero, self._tn_reward), zero)
            + torch.where(fp, torch.full_like(zero, self._fp_cost), zero)
            + torch.where(fn, torch.full_like(zero, self._fn_cost), zero)
        )
        conf_bonus = self._confidence_weight * gat_prob * attack_pred.float()

        components = {
            "r_classification": base,
            "r_agreement": zero,
            "r_confidence": conf_bonus,
            "r_disagreement_penalty": zero,
            "r_overconfidence_penalty": zero,
            "r_balance": zero,
        }
        total = sum(components.values())
        return total, components
