"""Fusion feature extractors for DQN state construction.

Each extractor knows how to derive a fixed-size feature vector from one
model's output.  Extractors are stateless and registered in the model
registry so that ``cache_predictions`` can iterate them generically.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class FusionFeatureExtractor(Protocol):
    """Extracts a fixed-size feature vector from a model's output for DQN fusion."""

    @property
    def feature_dim(self) -> int: ...

    @property
    def confidence_index(self) -> int:
        """Index of confidence feature within this extractor's output."""
        ...

    def extract(
        self,
        model: torch.nn.Module,
        graph,
        batch_idx: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor: ...


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

    def extract(
        self,
        model: torch.nn.Module,
        graph,
        batch_idx: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        edge_attr = (
            getattr(graph, "edge_attr", None) if getattr(model, "_uses_edge_attr", False) else None
        )
        cont, canid_logits, nbr_logits, z, _, _ = model(
            graph.x, graph.edge_index, batch_idx, edge_attr=edge_attr,
            node_id=graph.node_id,
        )
        recon_err = F.mse_loss(cont, graph.x, reduction="none").mean().item()
        canid_err = F.cross_entropy(canid_logits, graph.node_id).item()
        nbr_targets = model.create_neighborhood_targets(graph.node_id, graph.edge_index, batch_idx)
        nbr_err = F.binary_cross_entropy_with_logits(
            nbr_logits, nbr_targets, reduction="mean"
        ).item()
        z_mean, z_std = z.mean().item(), z.std().item()
        z_max, z_min = z.max().item(), z.min().item()
        vgae_conf = 1.0 / (1.0 + recon_err)

        return torch.tensor(
            [
                recon_err,
                nbr_err,
                canid_err,
                z_mean,
                z_std,
                z_max,
                z_min,
                vgae_conf,
            ]
        )


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

    def extract(
        self,
        model: torch.nn.Module,
        graph,
        batch_idx: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        logits, emb = model(graph, return_embedding=True)
        emb_mean = emb.mean().item()
        emb_std = emb.std().item() if emb.numel() > 1 else 0.0
        emb_max, emb_min = emb.max().item(), emb.min().item()

        probs = F.softmax(logits, dim=1)
        p0, p1 = probs[0, 0].item(), probs[0, 1].item()
        entropy = -(probs * (probs + 1e-8).log()).sum().item()
        gat_conf = max(0.0, min(1.0, 1.0 - entropy / math.log(2)))

        return torch.tensor([p0, p1, emb_mean, emb_std, emb_max, emb_min, gat_conf])
