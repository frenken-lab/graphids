"""Autoencoder-style losses as ``nn.Module`` with a uniform signature.

``VGAETaskLoss`` is the recon + canid + nbr + KL loss that pairs with
``GraphAutoencoderNeighborhood``'s 5-tuple forward output. Moving it
out of the Lightning module is what lets KD compose with it as a drop-in
wrapper (see :class:`graphids.core.losses.distillation.FeatureDistillation`).

Signature contract: autoencoder losses take ``(student_outputs, batch)``
where ``student_outputs`` is whatever tuple the student's forward returns.

``num_ids`` is populated lazily by ``VGAEModule._build()`` from
``datamodule.num_ids``, since the true dataset vocabulary size isn't
known at ``instantiate`` time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGAETaskLoss(nn.Module):
    """VGAE training loss: recon MSE + canid CE + neighborhood BCE + KL.

    ``forward(student_outputs, batch)`` unpacks the 5-tuple
    ``(cont_out, canid_logits, nbr_logits, z, kl_per_node)`` returned
    by ``GraphAutoencoderNeighborhood``. The ``z`` latent is read off
    ``student_outputs`` directly by ``FeatureDistillation`` when
    wrapping this loss.

    Aux heads are training-only — the test-time anomaly score is
    calibrated max-σ over (recon, Mahalanobis on μ, KL); canid/nbr
    are not part of scoring. Their job is to shape μ during training
    so that CAN-ID identity and neighborhood structure are encoded
    in the latent (where Mahalanobis can pick up density shifts).

    Reconstruction targets are the original (un-masked) ``batch.x``;
    masked nodes contribute the strong "predict-from-neighbors" signal,
    unmasked nodes contribute a weak regularizer — the F.mse_loss mean
    averages over both.
    """

    def __init__(
        self,
        *,
        kl_weight: float = 0.01,
        canid_weight: float = 0.1,
        nbr_weight: float = 0.05,
        k_neg: int = 32,
        num_ids: int = 0,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.canid_weight = canid_weight
        self.nbr_weight = nbr_weight
        self.k_neg = k_neg
        self.num_ids = num_ids  # populated by VGAEModule._build() from dm.num_ids
        # Populated each forward() so VGAEModule._training_step_inner can
        # log per-component telemetry to MLflow.
        self.last_recon: torch.Tensor | None = None
        self.last_canid: torch.Tensor | None = None
        self.last_nbr: torch.Tensor | None = None
        self.last_kl: torch.Tensor | None = None

    def forward(
        self,
        student_outputs: tuple,
        batch,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cont_out, canid_logits, nbr_logits, _z, kl_per_node = student_outputs

        recon = F.mse_loss(cont_out, batch.x)

        # Canid CE + nbr BCE on masked nodes only — for unmasked nodes the
        # model received node_id as input, so predicting it back is trivial
        # identity recovery that dilutes the gradient signal.
        if mask is not None and mask.any():
            canid = F.cross_entropy(canid_logits[mask], batch.node_id[mask])
            src_masked = mask[batch.edge_index[0]]
            nbr_edge_index = batch.edge_index[:, src_masked]
        else:
            canid = F.cross_entropy(canid_logits, batch.node_id)
            nbr_edge_index = batch.edge_index

        from graphids.core.models.autoencoder.vgae import VGAE

        nbr_loss = VGAE.neighborhood_loss_negsampled(
            nbr_logits,
            batch.node_id,
            nbr_edge_index,
            self.num_ids,
            k_neg=self.k_neg,
        )
        kl = kl_per_node.mean()

        total = recon + self.canid_weight * canid + self.nbr_weight * nbr_loss + self.kl_weight * kl
        if not torch.isfinite(total):
            raise ValueError(
                "VGAETaskLoss non-finite: "
                f"recon={recon.item():.6g} canid={canid.item():.6g} "
                f"nbr={nbr_loss.item():.6g} kl={kl.item():.6g} "
                f"weights=(canid={self.canid_weight},nbr={self.nbr_weight},"
                f"kl={self.kl_weight}) training={self.training}"
            )
        self.last_recon = recon.detach()
        self.last_canid = canid.detach()
        self.last_nbr = nbr_loss.detach()
        self.last_kl = kl.detach()
        return total
