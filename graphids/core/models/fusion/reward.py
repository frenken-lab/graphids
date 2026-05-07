"""Fusion reward calculator. Reads named keys from the feature TensorDict."""

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

    def normalize(self, td: TensorDict) -> TensorDict:
        """Clamp confidence keys to [0, 1]. Returns a shallow-cloned TD."""
        out = td.clone(recurse=False)
        out["vgae"] = td["vgae"].clone(recurse=False)
        out["gat"] = td["gat"].clone(recurse=False)
        out["vgae", "conf"] = td["vgae", "conf"].clamp(0.0, 1.0)
        out["gat", "conf"] = td["gat", "conf"].clamp(0.0, 1.0)
        return out

    def derive_scores(self, td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (anomaly_scores[N], gat_probs_pos[N])."""
        anomaly = (td["vgae", "errors"] * self._vgae_weights).sum(dim=1).clamp(0.0, 1.0)
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
