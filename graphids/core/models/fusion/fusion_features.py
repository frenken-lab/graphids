"""Fusion feature extractors and state layout for DQN/bandit state construction.

Each extractor derives a fixed-size feature matrix [B, D] from one model's
batched output. Module-level ``EXTRACTORS`` defines the canonical ordering
that determines the 15-D state layout trained DQN/bandit checkpoints expect.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


class FeatureLayout(NamedTuple):
    """Offset, dim, and confidence index of one extractor inside the state vector."""

    offset: int
    dim: int
    confidence_idx: int


class VGAEFusionExtractor:
    """Extract 8-D features from VGAE output.

    Layout:
        [0:3]  errors  (node recon, neighbor, canid)
        [3:7]  latent stats  (mean, std, max, min)
        [7]    confidence  (1 / (1 + recon_err))
    """

    feature_dim = 8
    confidence_index = 7

    def extract(self, model: torch.nn.Module, batch, device: torch.device) -> torch.Tensor:
        edge_attr = getattr(batch, "edge_attr", None) if getattr(model, "_uses_edge_attr", False) else None
        cont, canid_logits, nbr_logits, z, _, _ = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr, node_id=batch.node_id,
        )
        b = batch.batch

        # Per-graph MSE reconstruction error
        node_sq_err = (cont - batch.x).pow(2).mean(dim=1)  # [N]
        recon_err = scatter(node_sq_err, b, dim=0, reduce="mean")  # [B]

        # Per-graph cross-entropy for CAN ID prediction
        canid_ce = F.cross_entropy(canid_logits, batch.node_id, reduction="none")  # [N]
        canid_err = scatter(canid_ce, b, dim=0, reduce="mean")  # [B]

        # Per-graph neighbor prediction loss
        nbr_targets = model.create_neighborhood_targets(batch.node_id, batch.edge_index, b)
        nbr_bce = F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets, reduction="none").mean(dim=1)  # [N]
        nbr_err = scatter(nbr_bce, b, dim=0, reduce="mean")  # [B]

        # Per-graph latent stats
        z_mean = scatter(z.mean(dim=1), b, dim=0, reduce="mean")  # [B]
        z_std = scatter(z.std(dim=1), b, dim=0, reduce="mean")  # [B]
        z_max = scatter(z.max(dim=1).values, b, dim=0, reduce="max")  # [B]
        z_min = scatter(z.min(dim=1).values, b, dim=0, reduce="min")  # [B]

        conf = 1.0 / (1.0 + recon_err)  # [B]

        return torch.stack([recon_err, nbr_err, canid_err, z_mean, z_std, z_max, z_min, conf], dim=1)


class GATFusionExtractor:
    """Extract 7-D features from GAT output.

    Layout:
        [0:2]  class probabilities  (class 0, class 1)
        [2:6]  embedding stats  (mean, std, max, min)
        [6]    confidence  (1 - normalized entropy)
    """

    feature_dim = 7
    confidence_index = 6

    def extract(self, model: torch.nn.Module, batch, device: torch.device) -> torch.Tensor:
        logits, emb = model(batch, return_embedding=True)  # logits [B,2], emb [B, D] (post-pool)

        probs = F.softmax(logits, dim=1)  # [B, 2]
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)  # [B]
        conf = (1.0 - entropy / math.log(2)).clamp(0.0, 1.0)  # [B]

        # emb is already graph-level [B, D] after GAT's pool layer
        emb_mean = emb.mean(dim=1)   # [B]
        emb_std = emb.std(dim=1)     # [B]
        emb_max = emb.max(dim=1).values  # [B]
        emb_min = emb.min(dim=1).values  # [B]

        return torch.cat([probs, emb_mean.unsqueeze(1), emb_std.unsqueeze(1),
                          emb_max.unsqueeze(1), emb_min.unsqueeze(1), conf.unsqueeze(1)], dim=1)


# ---------------------------------------------------------------------------
# Registry — order is load-bearing, matches the 15-D layout baked into
# existing DQN/bandit checkpoints.
# ---------------------------------------------------------------------------

EXTRACTORS: dict[str, VGAEFusionExtractor | GATFusionExtractor] = {
    "vgae": VGAEFusionExtractor(),
    "gat": GATFusionExtractor(),
}


def _build_layout() -> tuple[dict[str, FeatureLayout], int]:
    layout: dict[str, FeatureLayout] = {}
    offset = 0
    for name, ext in EXTRACTORS.items():
        layout[name] = FeatureLayout(offset, ext.feature_dim, offset + ext.confidence_index)
        offset += ext.feature_dim
    return layout, offset


LAYOUT, STATE_DIM = _build_layout()
