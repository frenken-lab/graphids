"""PyG graph utilities — batch indexing and attack type extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def get_batch_index(g, device: torch.device) -> torch.Tensor:
    """Get batch index from graph, creating a single-graph default if absent."""
    import torch

    if hasattr(g, "batch") and g.batch is not None:
        return g.batch
    return torch.zeros(g.x.size(0), dtype=torch.long, device=device)


def graph_label(g) -> int:
    """Extract scalar label from a PyG Data object (handles 0-D and 1-D y)."""
    return g.y.item() if g.y.dim() == 0 else int(g.y[0].item())


def graph_attack_type(g, default: int | None = -1) -> int | None:
    """Get attack_type from a PyG graph, with backward-compat default.

    Old caches (pre-v2.0.0) lack the attack_type attribute.  This centralises
    the hasattr guard so callers don't scatter version-gating inline.
    """
    if hasattr(g, "attack_type") and g.attack_type is not None:
        return g.attack_type.item()
    return default
