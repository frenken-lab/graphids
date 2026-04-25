"""Autoencoder-style losses as ``nn.Module`` with a uniform signature.

``VGAETaskLoss`` absorbs the four-head reconstruction loss that used to be
inlined in ``VGAEModule._task_loss``. Moving it out of the Lightning
module is what lets KD compose with it as a drop-in wrapper
(see :class:`graphids.core.losses.distillation.FeatureDistillation`).

Signature contract: autoencoder losses take ``(student_outputs, batch)``
where ``student_outputs`` is whatever tuple the student's forward returns.
This differs from the classification loss signature
``(logits, labels, graph=None)`` — VGAE has no scalar logits, it has a
6-tuple of reconstruction heads + latent + KL.

``num_ids`` is populated lazily by ``VGAEModule.setup()`` from
``datamodule.num_ids``, since the true dataset vocabulary size isn't known
at ``instantiate`` time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGAETaskLoss(nn.Module):
    """VGAE reconstruction loss: recon MSE + CAN-ID CE + neighborhood + KL.

    ``forward(student_outputs, batch)`` unpacks the 5-tuple
    ``(cont_out, canid_logits, nbr_logits, z, kl_loss)`` returned by
    ``GraphAutoencoderNeighborhood``. The ``z`` latent is not used by the
    task loss itself but ``FeatureDistillation`` reads it off
    ``student_outputs`` directly when wrapping this loss.
    """

    def __init__(
        self,
        *,
        canid_weight: float = 0.1,
        nbr_weight: float = 0.05,
        kl_weight: float = 0.01,
        k_neg: int = 32,
        num_ids: int = 0,
    ):
        super().__init__()
        self.canid_weight = canid_weight
        self.nbr_weight = nbr_weight
        self.kl_weight = kl_weight
        self.k_neg = k_neg
        self.num_ids = num_ids  # populated by VGAEModule.setup() from dm.num_ids
        # Populated each forward() so VGAEModule._training_step_inner can log
        # per-component losses to MLflow. Recon-dominance is the failure
        # mode to watch after the sigmoid + masking deletions: if recon
        # collapses to ~0 while canid_weight*canid and nbr_weight*nbr stay
        # large, the encoder isn't learning structure.
        self.last_recon: torch.Tensor | None = None
        self.last_canid: torch.Tensor | None = None
        self.last_nbr: torch.Tensor | None = None
        self.last_kl: torch.Tensor | None = None

    def forward(self, student_outputs: tuple, batch) -> torch.Tensor:
        cont_out, canid_logits, nbr_logits, _z, kl_loss = student_outputs
        target = batch.x

        recon = F.mse_loss(cont_out, target)

        canid = F.cross_entropy(canid_logits, batch.node_id)

        # Lazy import breaks the circular dep between the model package and
        # the losses package. neighborhood_loss_negsampled is a pure tensor
        # op that happens to live as a classmethod on the autoencoder.
        from graphids.core.models.autoencoder.vgae import GraphAutoencoderNeighborhood

        nbr_loss = GraphAutoencoderNeighborhood.neighborhood_loss_negsampled(
            nbr_logits,
            batch.node_id,
            batch.edge_index,
            self.num_ids,
            k_neg=self.k_neg,
        )

        total = (
            recon
            + self.canid_weight * canid
            + self.nbr_weight * nbr_loss
            + self.kl_weight * kl_loss
        )
        if not torch.isfinite(total):
            raise ValueError(
                "VGAETaskLoss non-finite: "
                f"recon={recon.item():.6g} canid={canid.item():.6g} "
                f"nbr={nbr_loss.item():.6g} kl={kl_loss.item():.6g} "
                f"weights=(canid={self.canid_weight},nbr={self.nbr_weight},kl={self.kl_weight}) "
                f"training={self.training}"
            )
        self.last_recon = recon.detach()
        self.last_canid = canid.detach()
        self.last_nbr = nbr_loss.detach()
        self.last_kl = kl_loss.detach()
        return total
