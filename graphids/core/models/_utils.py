"""Shared utilities for model forward passes."""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv


class InputEncoder(nn.Module):
    """Shared input encoding for VGAE and GAT models.

    Encapsulates CAN ID embedding, optional feature projection, and the
    concatenation of embedding + continuous features. Both VGAE and GAT
    use identical input encoding logic.

    Attribute names (id_embedding, feat_proj) are preserved for clarity,
    though composing this as ``self.input_encoder`` changes state_dict
    key prefixes (acceptable on pre-sweep branch).
    """

    def __init__(
        self,
        num_ids: int,
        in_channels: int,
        embedding_dim: int,
        conv_type: str = "gat",
        edge_dim: int | None = None,
        proj_dim: int = 0,
    ):
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        self.num_ids = num_ids
        self.conv_type = conv_type
        self._uses_edge_attr = conv_type in ("transformer", "gatv2")
        self._edge_dim = edge_dim if self._uses_edge_attr else None
        self._proj_dim = proj_dim

        if proj_dim > 0:
            self.feat_proj = nn.Linear(in_channels - 1, proj_dim)
        else:
            self.feat_proj = None

        cont_dim = proj_dim if proj_dim > 0 else (in_channels - 1)
        self.out_dim = embedding_dim + cont_dim

    def forward(self, x: Tensor) -> Tensor:
        """Encode raw node features into model input.

        Args:
            x: Node features ``[num_nodes, in_channels]`` where ``x[:, 0]``
               is the CAN ID index and ``x[:, 1:]`` are continuous features.

        Returns:
            Encoded features ``[num_nodes, out_dim]``.
        """
        id_emb = self.id_embedding(x[:, 0].long())
        other_feats = x[:, 1:]
        if self.feat_proj is not None:
            other_feats = self.feat_proj(other_feats)
        return torch.cat([id_emb, other_feats], dim=1)


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
