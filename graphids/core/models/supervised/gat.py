from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    JumpingKnowledge,
    global_mean_pool,
)
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.utils import add_self_loops, remove_self_loops

from .._conv import InputEncoder, _make_conv, conv_forward, resolve_edge_dim


class GATWithJK(nn.Module):
    """Graph Attention Network with Jumping Knowledge connections.

    Supports GATConv (default), GATv2Conv, and TransformerConv via conv_type.
    TransformerConv natively uses edge_attr, enabling the 11-D edge features
    (frequency, temporal intervals, bidirectionality, degree products) that
    GATConv ignores.
    """

    def __init__(
        self,
        id_encoder,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        heads=4,
        dropout=0.2,
        num_fc_layers=3,
        use_checkpointing=False,
        conv_type="gat",
        edge_dim=None,
        pool_aggrs=("mean",),
        proj_dim=0,
    ):
        super().__init__()

        # Shared input encoding (ID encoder + optional projection)
        self.input_encoder = InputEncoder(
            id_encoder=id_encoder,
            in_channels=in_channels,
            conv_type=conv_type,
            edge_dim=edge_dim,
            proj_dim=proj_dim,
        )
        self.dropout = dropout
        self.use_checkpointing = use_checkpointing
        self.conv_type = conv_type
        self._uses_edge_attr = self.input_encoder._uses_edge_attr
        self._proj_dim = proj_dim

        # GATv2Conv runs remove_self_loops + add_self_loops inside its forward,
        # which creates edge-count-shaped intermediates inside the checkpoint region.
        # Under VRAM pressure the use_reentrant=False recompute produces a count
        # off by 1, triggering CheckpointError. Fix: pre-add self-loops once here
        # (same fill_value='mean' GATv2Conv uses) and disable inside each layer so
        # the checkpoint region sees a stable edge_index with no edge-count ops.
        self._prepend_self_loops = conv_type == "gatv2"
        conv_kwargs: dict = {"add_self_loops": False} if self._prepend_self_loops else {}

        # GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = self.input_encoder.out_dim if i == 0 else hidden_channels * heads
            self.convs.append(
                _make_conv(
                    conv_type,
                    in_dim,
                    hidden_channels,
                    heads=heads,
                    edge_dim=edge_dim if self._uses_edge_attr else None,
                    **conv_kwargs,
                )
            )

        self.jk = JumpingKnowledge(
            mode="lstm", channels=hidden_channels * heads, num_layers=num_layers
        )

        # Pooling — "lstm" JK outputs single-layer dim (not concatenated)
        jk_out_dim = hidden_channels * heads
        pool_aggrs = pool_aggrs or ("mean",)
        if len(pool_aggrs) > 1:
            self.pool = MultiAggregation(list(pool_aggrs))
            fc_input_dim = jk_out_dim * len(pool_aggrs)
        else:
            self.pool = None  # use global_mean_pool
            fc_input_dim = jk_out_dim

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        for _ in range(num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(fc_input_dim, fc_input_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(p=dropout))
        self.fc_layers.append(nn.Linear(fc_input_dim, out_channels))

    @classmethod
    def from_config(cls, cfg, id_encoder, in_ch: int) -> GATWithJK:
        """Construct from a config."""
        conv_type = getattr(cfg, "conv_type", "gatv2")
        edge_dim = getattr(cfg, "edge_dim", 11)
        return cls(
            id_encoder=id_encoder,
            in_channels=in_ch,
            hidden_channels=cfg.hidden,
            out_channels=cfg.num_classes,
            num_layers=cfg.layers,
            heads=cfg.heads,
            dropout=getattr(cfg, "dropout", 0.2),
            num_fc_layers=cfg.fc_layers,
            conv_type=conv_type,
            edge_dim=resolve_edge_dim(conv_type, edge_dim),
            pool_aggrs=getattr(cfg, "pool_aggrs", None),
            proj_dim=getattr(cfg, "proj_dim", 0),
            use_checkpointing=getattr(cfg, "gradient_checkpointing", True),
        )

    def _pool(self, x, batch):
        if self.pool is not None:
            return self.pool(x, batch)
        return global_mean_pool(x, batch)

    def forward(
        self,
        data,
        return_intermediate=False,
        return_attention_weights=False,
        return_embedding=False,
    ):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None) if self._uses_edge_attr else None
        node_id = data.node_id

        x = self.input_encoder(x, node_id)

        # Pre-add self-loops once outside the checkpoint region.  Each GATv2Conv
        # layer was constructed with add_self_loops=False so it won't touch the
        # edge structure again — the checkpoint region is edge-count-stable.
        if self._prepend_self_loops:
            num_nodes = x.size(0)
            if edge_attr is not None:
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value="mean", num_nodes=num_nodes
                )
            else:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        attention_weights = [] if return_attention_weights else None

        xs = []
        for conv in self.convs:
            if return_attention_weights and self.conv_type == "gat":
                # Attention weight extraction requires direct conv call
                x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                attention_weights.append(alpha.detach().cpu())
            else:
                x = conv_forward(
                    conv,
                    x,
                    edge_index,
                    edge_attr,
                    batch=batch,
                    dropout_p=self.dropout,
                    training=self.training,
                    use_checkpointing=self.use_checkpointing,
                )
            xs.append(x)
        if return_attention_weights:
            return xs, attention_weights
        if return_intermediate:
            return xs
        x = self.jk(xs)
        x = self._pool(x, batch)
        if return_embedding:
            emb = x.clone()
            for layer in self.fc_layers:
                x = layer(x)
            return x, emb
        for layer in self.fc_layers:
            x = layer(x)
        return x
