"""Autoencoder-style losses as ``nn.Module`` with a uniform signature.

``VGAETaskLoss`` is the recon + KL loss that pairs with
``GraphAutoencoderNeighborhood``'s 3-tuple forward output. Moving it
out of the Lightning module is what lets KD compose with it as a drop-in
wrapper (see :class:`graphids.core.losses.distillation.FeatureDistillation`).

Signature contract: autoencoder losses take ``(student_outputs, batch)``
where ``student_outputs`` is whatever tuple the student's forward returns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGAETaskLoss(nn.Module):
    """VGAE reconstruction + KL loss.

    ``forward(student_outputs, batch)`` unpacks the 3-tuple
    ``(cont_out, z, kl_per_node)`` returned by
    ``GraphAutoencoderNeighborhood``. ``z`` is read off
    ``student_outputs`` directly by ``FeatureDistillation`` when
    wrapping this loss.

    Reconstruction targets are the original (un-masked) ``batch.x``;
    masked nodes contribute the strong "predict-from-neighbors" signal,
    unmasked nodes contribute a weak regularizer — the F.mse_loss mean
    averages over both.
    """

    def __init__(self, *, kl_weight: float = 0.01):
        super().__init__()
        self.kl_weight = kl_weight
        # Populated each forward() so VGAEModule._training_step_inner can
        # log per-component telemetry to MLflow.
        self.last_recon: torch.Tensor | None = None
        self.last_kl: torch.Tensor | None = None

    def forward(self, student_outputs: tuple, batch) -> torch.Tensor:
        cont_out, _z, kl_per_node = student_outputs
        recon = F.mse_loss(cont_out, batch.x)
        kl = kl_per_node.mean()
        total = recon + self.kl_weight * kl
        if not torch.isfinite(total):
            raise ValueError(
                "VGAETaskLoss non-finite: "
                f"recon={recon.item():.6g} kl={kl.item():.6g} "
                f"weights=(kl={self.kl_weight}) training={self.training}"
            )
        self.last_recon = recon.detach()
        self.last_kl = kl.detach()
        return total
