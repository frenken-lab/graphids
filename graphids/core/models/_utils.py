"""Shared utilities for model forward passes."""

from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv


def _make_conv(
    conv_type: str, in_dim: int, out_dim: int, heads: int, edge_dim: int | None = None, **kwargs
):
    """Factory for graph attention convolution layers."""
    if conv_type == "transformer":
        return TransformerConv(
            in_dim, out_dim, heads=heads, edge_dim=edge_dim, concat=True, **kwargs
        )
    elif conv_type == "gatv2":
        return GATv2Conv(in_dim, out_dim, heads=heads, edge_dim=edge_dim, concat=True, **kwargs)
    else:
        return GATConv(in_dim, out_dim, heads=heads, concat=True, **kwargs)


def checkpoint_conv(conv, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
    """Run a graph conv layer through gradient checkpointing.

    Uses default-arg capture to avoid the stale-closure bug in loops.
    """
    if edge_attr is not None:
        return checkpoint(
            lambda xi, c=conv, ei=edge_index, ea=edge_attr: c(xi, ei, ea), x, use_reentrant=False
        )
    return checkpoint(lambda xi, c=conv, ei=edge_index: c(xi, ei), x, use_reentrant=False)
