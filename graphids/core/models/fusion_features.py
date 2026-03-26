"""Fusion feature extractors for DQN state construction.

Each extractor derives a fixed-size feature matrix [B, D] from one model's
batched output. Extractors are stateless and registered in the model registry
so that ``FusionDataModule.cache_predictions`` can iterate them generically.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


@runtime_checkable
class FusionFeatureExtractor(Protocol):
    """Extracts a [B, D] feature matrix from a model's batched output."""

    @property
    def feature_dim(self) -> int: ...

    @property
    def confidence_index(self) -> int:
        """Index of confidence feature within this extractor's output."""
        ...

    def extract(
        self,
        model: torch.nn.Module,
        batch,
        device: torch.device,
    ) -> torch.Tensor:
        """Return [B, feature_dim] tensor from a batched PyG Data object."""
        ...


class VGAEFusionExtractor:
    """Extract 8-D features from VGAE output.

    Layout:
        [0:3]  errors  (node recon, neighbor, canid)
        [3:7]  latent stats  (mean, std, max, min)
        [7]    confidence  (1 / (1 + recon_err))
    """

    @property
    def feature_dim(self) -> int:
        return 8

    @property
    def confidence_index(self) -> int:
        return 7

    def extract(self, model: torch.nn.Module, batch, device: torch.device) -> torch.Tensor:
        edge_attr = getattr(batch, "edge_attr", None) if getattr(model, "_uses_edge_attr", False) else None
        cont, canid_logits, nbr_logits, z, _, _ = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr, node_id=batch.node_id,
        )
        B = int(batch.batch.max()) + 1
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

    @property
    def feature_dim(self) -> int:
        return 7

    @property
    def confidence_index(self) -> int:
        return 6

    def extract(self, model: torch.nn.Module, batch, device: torch.device) -> torch.Tensor:
        logits, emb = model(batch, return_embedding=True)  # logits [B,2], emb [N, D]
        b = batch.batch

        probs = F.softmax(logits, dim=1)  # [B, 2]
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)  # [B]
        conf = (1.0 - entropy / math.log(2)).clamp(0.0, 1.0)  # [B]

        emb_mean = scatter(emb.mean(dim=1), b, dim=0, reduce="mean")  # [B]
        emb_std = scatter(emb.std(dim=1), b, dim=0, reduce="mean")  # [B]
        emb_max = scatter(emb.max(dim=1).values, b, dim=0, reduce="max")  # [B]
        emb_min = scatter(emb.min(dim=1).values, b, dim=0, reduce="min")  # [B]

        return torch.cat([probs, emb_mean.unsqueeze(1), emb_std.unsqueeze(1),
                          emb_max.unsqueeze(1), emb_min.unsqueeze(1), conf.unsqueeze(1)], dim=1)
