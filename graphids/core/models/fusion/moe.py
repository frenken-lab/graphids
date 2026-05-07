"""MoE+BCE per-sample gated fusion: K experts with softmax router, dense soft-gated.

Implements the canonical Jacobs & Jordan (1991) "Adaptive Mixtures of Local
Experts" formulation: every sample passes through every expert; the gate
emits per-sample weights ``w(x) ∈ Δ^{K-1}``; final prediction is the convex
combination ``Σᵢ wᵢ(x) · sigmoid(hᵢ(x))``. Trained end-to-end with BCE on
the mixed score — no per-expert supervision, no auxiliary losses in v0.

Why dense soft-gated and not sparse top-k: sparse routing's value is
conditional compute at scale (Switch Transformer, Mixtral). At K=3 with
18-dim features the FLOPs argument is moot, and soft blending is the
hypothesis we want to test. Design rationale, variant survey, and
escalation paths: ``docs/drafts/moe-fusion-design.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from .base import FusionModuleBase, flatten_features


def _build_head(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """[Linear → ReLU → Dropout(0.2)] × len(hidden), then Linear(out_dim).

    Same body shape as ``MLPFusionModule`` so each expert has comparable
    capacity to the supervised baseline it must beat.
    """
    layers: list[nn.Module] = []
    cur = in_dim
    for h in hidden:
        layers.extend([nn.Linear(cur, h), nn.ReLU(), nn.Dropout(0.2)])
        cur = h
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


class MoEFusionModule(FusionModuleBase):
    """Dense soft-gated mixture of K identical experts over the flat feature vector.

    Specialization is emergent: experts share architecture and input;
    only the gate's softmax over per-sample logits selects how their
    outputs combine. If gate entropy stays at ``log(K)`` (uniform) on
    a fitted run, the features carry no routable signal — see
    diagnostics + escalation table in the design doc.
    """

    automatic_optimization = True

    def __init__(
        self,
        state_dim: int = 18,
        num_experts: int = 3,
        expert_hidden: tuple[int, ...] = (64, 32),
        gate_hidden: tuple[int, ...] = (32,),
        lr: float = 1e-3,
        decision_threshold: float = 0.5,
    ):
        super().__init__(state_dim=state_dim, decision_threshold=decision_threshold)
        self._store_init_kwargs(locals())

        self.experts = nn.ModuleList(
            [_build_head(state_dim, expert_hidden, out_dim=1) for _ in range(num_experts)]
        )
        self.gate = _build_head(state_dim, gate_hidden, out_dim=num_experts)

        # Last-batch routing diagnostics; set by forward_scores, read by
        # training_step / validation_step. Never read across batches.
        self._last_gate_weights: torch.Tensor | None = None
        self._last_expert_scores: torch.Tensor | None = None

    def forward_scores(self, td: TensorDict) -> torch.Tensor:
        x = flatten_features(td).to(self.device)
        # [N, K] expert scores in (0, 1); sigmoid because the mix output is
        # also in (0, 1) and a convex combination of probs stays a prob.
        expert_scores = torch.sigmoid(torch.cat([h(x) for h in self.experts], dim=-1))
        weights = torch.softmax(self.gate(x), dim=-1)  # [N, K]

        self._last_gate_weights = weights.detach()
        self._last_expert_scores = expert_scores.detach()

        mixed = (expert_scores * weights).sum(-1)
        return mixed.clamp(1e-7, 1 - 1e-7)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    # -- routing diagnostics -------------------------------------------------

    def _log_gate_diagnostics(self, prefix: str) -> None:
        w = self._last_gate_weights
        s = self._last_expert_scores
        if w is None or s is None:
            return
        # Entropy: 0 = collapsed to one expert, log(K) = uniform routing.
        entropy = -(w * w.clamp_min(1e-9).log()).sum(-1).mean()
        self.log(f"{prefix}/gate_entropy", entropy.item())
        # Per-expert utilization: mean weight assigned across the batch.
        # Detects dead experts (one usage → 0).
        mean_w = w.mean(dim=0)
        for i in range(mean_w.numel()):
            self.log(f"{prefix}/expert_usage_{i}", mean_w[i].item())
        # Expert disagreement: if experts are duplicates, var → 0 and
        # gating is meaningless even at high entropy.
        self.log(f"{prefix}/expert_disagreement", s.var(dim=-1).mean().item())

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self._log_gate_diagnostics("train")
        return loss

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        self._log_gate_diagnostics("val")
