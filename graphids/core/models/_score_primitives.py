"""Stateless anomaly-score primitives — closed-form, no learned params.

Each function here produces a per-node or per-graph anomaly signal from
graph data alone (no encoder weights, no calibration). They're called
from :class:`graphids.core.models.autoencoder.vgae.VGAE._score`, then
calibrated and combined via the max-σ aggregator. Lifted out of the
VGAE body so the model class stays focused on architecture; new
primitives drop in here without touching the model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


def tam_affinity(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Per-node ``1 - mean cos(z_v, z_u)`` over outgoing edges (Qiao & Pang, NeurIPS 2023). Range ``[0, 2]``."""
    src, dst = edge_index
    sim = F.cosine_similarity(z[src], z[dst], dim=-1)
    per_node = scatter(sim, src, dim=0, dim_size=z.size(0), reduce="mean")
    return 1.0 - per_node


def rayleigh_quotient(
    x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-graph Rayleigh quotient of node features w.r.t. the graph Laplacian.

    ``RQ(X) = Σ_(i,j)∈E ||x_i - x_j||² / ||X||_F²`` — the edge form of
    ``tr(Xᵀ L X) / ||X||_F²`` for the unnormalized Laplacian, avoiding
    materializing ``L``. Measures feature smoothness on the graph: low
    when neighbors agree, high when one node breaks the local pattern.
    Closed-form, no training (Dong et al., RQGNN ICLR 2024).
    """
    src, dst = edge_index
    diff_sq = (x[src] - x[dst]).pow(2).sum(dim=-1)  # [E]
    norm_sq = x.pow(2).sum(dim=-1)  # [N]
    if batch is None:
        return diff_sq.sum() / (norm_sq.sum() + 1e-8)
    edge_batch = batch[src]
    num = scatter(diff_sq, edge_batch, dim=0, reduce="sum")
    denom = scatter(norm_sq, batch, dim=0, reduce="sum")
    return num / (denom + 1e-8)
