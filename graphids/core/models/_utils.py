"""Shared utilities for model forward passes."""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, GATv2Conv, GPSConv, TransformerConv
from torch_geometric.nn.norm import GraphNorm


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
        self._uses_edge_attr = conv_type in ("transformer", "gatv2", "gps")
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


class _ProjectedGPS(nn.Module):
    """Linear projection → GPSConv. Bridges dimension mismatch for GPS residuals."""

    def __init__(self, in_dim: int, channels: int, gps: GPSConv):
        super().__init__()
        self.proj = nn.Linear(in_dim, channels)
        self.gps = gps
        # Expose attributes that conv_forward and build_conv_stack inspect
        self.in_channels = in_dim
        self.out_channels = channels
        self.heads = 1  # output is already channels-wide

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None,
                batch: Tensor | None = None) -> Tensor:
        x = self.proj(x)
        return self.gps(x, edge_index, batch=batch, edge_attr=edge_attr)


def _make_conv(
    conv_type: str, in_dim: int, out_dim: int, heads: int, edge_dim: int | None = None, **kwargs
):
    """Factory for graph convolution layers.

    Supports: gat, gatv2, transformer, gps.
    GPS wraps a GATv2Conv with global self-attention (Rampasek et al., 2022).
    """
    if conv_type == "transformer":
        return TransformerConv(
            in_dim, out_dim, heads=heads, edge_dim=edge_dim, concat=True, **kwargs
        )
    elif conv_type == "gps":
        # GPS requires in_dim == out_dim for residual connections.
        # Inner GATv2Conv uses concat=False to preserve dimensionality.
        channels = out_dim * heads
        inner = GATv2Conv(
            channels, channels, heads=heads, concat=False, edge_dim=edge_dim,
        )
        gps = GPSConv(channels, inner, heads=heads, attn_type="multihead", dropout=0.1)
        if in_dim != channels:
            return _ProjectedGPS(in_dim, channels, gps)
        return gps
    elif conv_type == "gatv2":
        return GATv2Conv(in_dim, out_dim, heads=heads, edge_dim=edge_dim, concat=True, **kwargs)
    else:
        return GATConv(in_dim, out_dim, heads=heads, concat=True, **kwargs)


def build_conv_stack(
    conv_type: str,
    in_dim: int,
    target_dims: list[int],
    edge_dim: int | None,
    heads_first: int = 1,
    batch_norm: bool = True,
) -> tuple[nn.ModuleList, nn.ModuleList]:
    """Build a stack of graph conv layers with optional GraphNorm.

    Resolves multi-head output dimensions: if target_dim is divisible by heads,
    uses target_dim // heads per head; otherwise falls back to heads=1.

    Returns (conv_layers, norm_layers). norm_layers may be shorter than
    conv_layers if batch_norm is False.
    """
    convs = nn.ModuleList()
    norms = nn.ModuleList()
    for i, target_dim in enumerate(target_dims):
        heads = heads_first if i == 0 else 1
        if heads > 1 and target_dim % heads == 0:
            out_per_head = target_dim // heads
        else:
            heads = 1
            out_per_head = target_dim
        convs.append(_make_conv(conv_type, in_dim, out_per_head, heads=heads, edge_dim=edge_dim))
        if batch_norm:
            norms.append(GraphNorm(out_per_head * heads))
        in_dim = out_per_head * heads
    return convs, norms


def checkpoint_conv(conv, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
    """Run a graph conv layer through gradient checkpointing.

    Uses default-arg capture to avoid the stale-closure bug in loops.
    """
    if edge_attr is not None:
        return checkpoint(
            lambda xi, c=conv, ei=edge_index, ea=edge_attr: c(xi, ei, ea), x, use_reentrant=False
        )
    return checkpoint(lambda xi, c=conv, ei=edge_index: c(xi, ei), x, use_reentrant=False)


def conv_forward(
    conv,
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor | None = None,
    bn: nn.Module | None = None,
    batch: Tensor | None = None,
    activation=torch.nn.functional.relu,
    dropout_p: float = 0.0,
    training: bool = True,
    use_checkpointing: bool = False,
) -> Tensor:
    """Apply a graph conv layer with optional norm, activation, and dropout.

    Args:
        conv: Graph convolution layer (GATConv, GATv2Conv, TransformerConv).
        x: Node features ``[num_nodes, in_dim]``.
        edge_index: Edge indices ``[2, num_edges]``.
        edge_attr: Optional edge features.
        bn: Optional norm layer (GraphNorm or BatchNorm1d).
        batch: Batch assignment vector for GraphNorm. Ignored by BatchNorm1d.
        activation: Activation function (default: F.relu). Pass None to skip.
        dropout_p: Dropout probability (0.0 = no dropout).
        training: Whether in training mode (affects dropout).
        use_checkpointing: Use gradient checkpointing for the conv layer.

    Returns:
        Transformed node features.
    """
    _is_gps = isinstance(conv, (GPSConv, _ProjectedGPS))
    if use_checkpointing and x.requires_grad and not _is_gps:
        x = checkpoint_conv(conv, x, edge_index, edge_attr)
    elif _is_gps:
        x = conv(x, edge_index, edge_attr=edge_attr, batch=batch)
    else:
        x = conv(x, edge_index, edge_attr) if edge_attr is not None else conv(x, edge_index)
    # GPS has internal norm, activation, and dropout — skip external ones
    if _is_gps:
        return x
    if bn is not None:
        x = bn(x, batch) if isinstance(bn, GraphNorm) else bn(x)
    if activation is not None:
        x = activation(x)
    if dropout_p > 0.0:
        x = torch.nn.functional.dropout(x, p=dropout_p, training=training)
    return x
