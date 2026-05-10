"""Autoencoder-style losses with a uniform signature."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGAETaskLoss(nn.Module):
    """VGAE training loss: recon MSE + canid CE + KL + edge_attr recon + GAD-NR nbr.

    ``forward(student_outputs, batch)`` unpacks the 6-tuple
    ``(cont_out, canid_logits, nbr_pred, z, kl_per_node, edge_logits)``
    returned by ``VGAE``. The ``z`` latent is read off ``student_outputs``
    directly by ``FeatureDistillation`` when wrapping this loss.

    Aux heads are training-only — the test-time anomaly score is
    calibrated max-σ over (recon, recon_max, TAM affinity, RQ); canid
    and nbr are not part of scoring. canid shapes μ so CAN-ID identity
    is encoded in the latent; nbr (GAD-NR, Roy et al. WSDM 2024) shapes
    z so neighbor-mean is predictable from each node's own latent.

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
        nbr_weight: float = 0.1,
        edge_weight: float = 0.1,
        k_neg: int = 32,
        num_ids: int = 0,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.canid_weight = canid_weight
        self.nbr_weight = nbr_weight
        self.edge_weight = edge_weight
        self.k_neg = k_neg
        self.num_ids = num_ids  # populated by VGAEModule._build() from dm.num_ids
        # Populated each forward() so VGAE.training_step can
        # log per-component telemetry to MLflow.
        self.last_recon: torch.Tensor | None = None
        self.last_canid: torch.Tensor | None = None
        self.last_nbr: torch.Tensor | None = None
        self.last_kl: torch.Tensor | None = None
        self.last_edge: torch.Tensor | None = None

    def forward(
        self,
        student_outputs: tuple,
        batch,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cont_out, canid_logits, nbr_pred, z, kl_per_node, edge_logits = student_outputs

        recon = F.mse_loss(cont_out, batch.x)

        if mask is not None and mask.any():
            canid = F.cross_entropy(canid_logits[mask], batch.node_id[mask])
        else:
            canid = F.cross_entropy(canid_logits, batch.node_id)

        kl = kl_per_node.mean()

        # GAD-NR neighborhood loss. Empirical target = per-source-node mean of
        # neighbor latents; clamp the count to avoid 0/0 NaN on isolated source
        # nodes (matches the `_per_graph_masked_recon` pattern).
        from torch_geometric.utils import scatter

        src, dst = batch.edge_index
        n_nodes = z.size(0)
        sum_z = scatter(z[dst], src, dim=0, dim_size=n_nodes, reduce="sum")
        count = (
            scatter(
                torch.ones(src.size(0), device=z.device, dtype=z.dtype),
                src,
                dim=0,
                dim_size=n_nodes,
                reduce="sum",
            )
            .unsqueeze(-1)
            .clamp(min=1.0)
        )
        nbr_targets = sum_z / count
        nbr_loss = kl_neighbor_loss(nbr_pred, nbr_targets)

        # Edge-attribute reconstruction. Only present when the model's conv
        # stack consumes edge_attr (otherwise edge_logits is None and the
        # batch's edge_attr never entered the latent — supervising it would
        # make the head learn from random noise).
        edge_attr = getattr(batch, "edge_attr", None)
        if edge_logits is not None and edge_attr is not None:
            edge_recon = F.mse_loss(edge_logits, edge_attr)
        else:
            edge_recon = recon.new_zeros(())

        total = (
            recon
            + self.canid_weight * canid
            + self.nbr_weight * nbr_loss
            + self.kl_weight * kl
            + self.edge_weight * edge_recon
        )
        if not torch.isfinite(total):
            raise ValueError(
                "VGAETaskLoss non-finite: "
                f"recon={recon.item():.6g} canid={canid.item():.6g} "
                f"nbr={nbr_loss.item():.6g} kl={kl.item():.6g} "
                f"edge={edge_recon.item():.6g} "
                f"weights=(canid={self.canid_weight},nbr={self.nbr_weight},"
                f"kl={self.kl_weight},edge={self.edge_weight}) "
                f"training={self.training}"
            )
        self.last_recon = recon.detach()
        self.last_canid = canid.detach()
        self.last_nbr = nbr_loss.detach()
        self.last_kl = kl.detach()
        self.last_edge = edge_recon.detach()
        return total


def kl_neighbor_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask_len: int | None = None,
) -> torch.Tensor:
    r"""KL divergence between two empirical multivariate Gaussians fit to node-latent sets.

    GAD-NR (Roy et al., "GAD-NR: Graph Anomaly Detection via Neighborhood
    Reconstruction", WSDM 2024 — arXiv:2306.01951) trains a per-node
    neighborhood predictor whose decoder outputs the parameters of a Gaussian
    over its expected neighbor latents; the loss is the KL divergence between
    that prediction and the empirical Gaussian fit to the actual neighbors.

    This is the *replacement* for the deleted ``neighborhood_decoder`` in
    ``VGAE._build`` — the deleted version predicted a categorical bag over
    the CAN-ID vocabulary (vocab-bound, UNK-cliff on OOD, 1791-dim matmul
    that hit V100 fp32 overflow). GAD-NR's mechanism is vocab-free and
    operates entirely in latent space; sibling concept, different mechanism.

    Closed-form multivariate Gaussian KL (textbook convention, NOT the
    GAD-NR notebook / pygod sign — see "Differences" below):

    .. math::
        \mathrm{KL}(\mathcal{N}_1 \,\|\, \mathcal{N}_2)
        = \tfrac{1}{2}\Big[
            \log\frac{\det\Sigma_2}{\det\Sigma_1} - d
            + \mathrm{tr}(\Sigma_2^{-1}\Sigma_1)
            + (\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1)
        \Big]

    Both inputs are ``[N, d]`` tensors. Each is fit with sample mean and
    sample-covariance + I (the identity regularizer prevents singular Σ
    when N is small or features are collinear).

    Cross-references
    ----------------
    - Paper: Roy et al. arXiv:2306.01951 (WSDM 2024).
    - Reference repo: https://github.com/Graph-COM/GAD-NR
    - pygod port: ``pygod.nn.functional.KL_neighbor_loss``
      (https://github.com/pygod-team/pygod/blob/main/pygod/nn/functional.py)

    Differences from the pygod port
    -------------------------------
    - **Gradient flow.** pygod calls ``.cpu().detach()`` on both inputs,
      which silently makes the function non-differentiable. We omit that —
      this version supports both training-loss usage (gradients flow) and
      inference-score usage (call site wraps in ``torch.no_grad()``).
    - **Type cleanliness.** pygod uses ``math.log`` on a torch tensor,
      relying on implicit ``__float__`` coercion. We use ``torch.log``.
    - **Sign convention (CORRECTED).** Both pygod and the original
      GAD-NR notebook (Graph-COM/GAD-NR/GAD-NR_inj_cora.ipynb) use
      ``log(det Σ_1 / det Σ_2)`` — the inverse of the textbook KL form.
      That formula can be negative (verified empirically:
      produced ``last_nbr ≈ −5.96`` in our smoke run) and so is not a
      proper KL divergence; minimizing it rewards the encoder for
      collapsing prediction variance. We use the standard form
      ``log(det Σ_2 / det Σ_1)`` instead, which is non-negative and
      matches what KL minimization is supposed to do.
    - **W2 sibling.** pygod also exposes ``W2_neighbor_loss``; that
      implementation has a bare-newline bug (``+ torch.trace(...)`` parses
      as a no-op unary plus on a separate line, dropping the trace term).
      Not ported here.

    Parameters
    ----------
    predictions : torch.Tensor of shape ``[N, d]``
        Model-predicted neighbor latents (decoder output) — first set.
    targets : torch.Tensor of shape ``[N, d]``
        Empirical neighbor latents — second set.
    mask_len : int, optional
        If given, restrict both inputs to their first ``mask_len`` rows
        (matches pygod's masking interface).

    Returns
    -------
    torch.Tensor
        Scalar (0-dim) KL divergence.
    """
    if mask_len is not None:
        predictions = predictions[:mask_len, :]
        targets = targets[:mask_len, :]

    n, d = predictions.shape
    eye = torch.eye(d, device=predictions.device, dtype=predictions.dtype)

    mean_p = predictions.mean(dim=0)
    mean_t = targets.mean(dim=0)
    diff_p = predictions - mean_p
    diff_t = targets - mean_t
    denom = max(n - 1, 1)
    cov_p = diff_p.T @ diff_p / denom + eye
    cov_t = diff_t.T @ diff_t / denom + eye

    cov_t_inv = torch.linalg.inv(cov_t)
    mu_diff = mean_t - mean_p

    log_det_term = torch.log(torch.det(cov_t) / torch.det(cov_p))
    trace_term = torch.trace(cov_t_inv @ cov_p)
    maha_term = mu_diff @ cov_t_inv @ mu_diff
    return 0.5 * (log_det_term - d + trace_term + maha_term)
