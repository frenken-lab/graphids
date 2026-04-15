from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._conv import (
    InputEncoder,
    build_conv_stack,
    build_encoder_stack,
    conv_forward,
    resolve_edge_dim,
)


class GraphAutoencoderNeighborhood(nn.Module):
    """
    Graph Autoencoder that reconstructs node features and edge list.

    This implementation follows a *progressive compression schedule* defined by
    `hidden_dims`, which is a list like [256, 128, 96, 48]. The last element is
    typically the `latent_dim` and is *not* used as a GAT output size; instead
    the encoder builds GAT layers targeting the preceding entries and the final
    latent `z` is produced via linear heads mapping the last encoder output to
    `latent_dim`.

    Decoder is the mirror of the encoder (reverse of the progressive schedule)
    and the final decoder GAT produces the reconstructed continuous features.

    Args (key):
      - num_ids: number of CAN ID tokens
      - in_channels: input channel count (including CAN ID as first column)
      - hidden_dims: compression schedule, e.g., [256,128,96,48] (last element is latent_dim)
      - latent_dim: dimensionality of latent `z` (if None and hidden_dims provided, inferred as hidden_dims[-1])
      - encoder_heads: number of heads for the first encoder layer (others default to 1)
      - decoder_heads: number of heads for decoder intermediate layers
      - embedding_dim: CAN ID embedding size
      - dropout: dropout probability
      - mlp_hidden: hidden dimension for neighborhood decoder MLP (if None, uses latent_dim)
    """

    def __init__(
        self,
        num_ids,
        in_channels,
        hidden_dims=None,
        latent_dim=32,
        encoder_heads=4,
        decoder_heads=4,
        embedding_dim=8,
        dropout=0.35,
        batch_norm=True,
        mlp_hidden=None,
        use_checkpointing=False,
        conv_type="gat",
        edge_dim=None,
        proj_dim=0,
        variational=True,
    ):
        super().__init__()

        # Shared input encoding (ID embedding + optional projection)
        self.input_encoder = InputEncoder(
            num_ids=num_ids,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            conv_type=conv_type,
            edge_dim=edge_dim,
            proj_dim=proj_dim,
        )
        self.num_ids = num_ids
        self.dropout_rate = dropout
        self.batch_norm = batch_norm
        self.use_checkpointing = use_checkpointing
        self.conv_type = conv_type
        self._uses_edge_attr = self.input_encoder._uses_edge_attr
        self._edge_dim = self.input_encoder._edge_dim
        self._proj_dim = proj_dim

        # Encoder conv stack (shared with DGI)
        gat_in_dim = self.input_encoder.out_dim
        self.gat_in_dim = gat_in_dim
        self.encoder_layers, self.encoder_bns, self.latent_in_dim = build_encoder_stack(
            hidden_dims,
            latent_dim,
            gat_in_dim,
            conv_type,
            self._edge_dim,
            encoder_heads=encoder_heads,
            batch_norm=batch_norm,
        )
        self.variational = variational
        self.z_mean = nn.Linear(self.latent_in_dim, latent_dim)
        if variational:
            self.z_logvar = nn.Linear(self.latent_in_dim, latent_dim)

        # Decoder: mirror of encoder, final layer outputs continuous features
        # Recompute encoder_targets (same logic as build_encoder_stack)
        if hidden_dims is not None and len(hidden_dims) >= 2 and hidden_dims[-1] == latent_dim:
            encoder_targets = hidden_dims[:-1]
        else:
            encoder_targets = hidden_dims if hidden_dims else [max(128, latent_dim * 2), latent_dim]
        decoder_targets = list(reversed(encoder_targets))
        # Replace last target with in_channels for reconstruction output
        decoder_targets[-1] = in_channels
        self.decoder_layers, self.decoder_bns = build_conv_stack(
            conv_type,
            latent_dim,
            decoder_targets,
            self._edge_dim,
            heads_first=decoder_heads,
            batch_norm=batch_norm,
        )
        # Remove the batch norm for the last decoder layer (sigmoid output, no BN)
        if batch_norm and len(self.decoder_bns) == len(decoder_targets):
            self.decoder_bns = self.decoder_bns[:-1]

        # CAN ID classifier head
        self.canid_classifier = nn.Linear(latent_dim, num_ids)

        # Neighborhood decoder MLP: use mlp_hidden if provided, else default to latent_dim for parameter efficiency
        if mlp_hidden is None:
            mlp_hidden = latent_dim  # Default to latent_dim for compact models
        self.neighborhood_decoder = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_ids),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.latent_dim = latent_dim

    def encode(self, x, edge_index, edge_attr=None, batch=None, node_id=None):
        x = self.input_encoder(x, node_id)
        for i, conv in enumerate(self.encoder_layers):
            bn = self.encoder_bns[i] if self.batch_norm else None
            x = conv_forward(
                conv,
                x,
                edge_index,
                edge_attr,
                bn=bn,
                batch=batch,
                dropout_p=self.dropout_rate,
                training=self.training,
                use_checkpointing=self.use_checkpointing,
            )
        mu = self.z_mean(x)
        if self.variational:
            logvar = self.z_logvar(x).clamp(-20, 20)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            z = mu
            kl_loss = mu.new_tensor(0.0)
        return z, kl_loss

    def decode_node(self, z, edge_index, edge_attr=None, batch=None):
        assert z.size(-1) == self.latent_dim, (
            f"Expected {self.latent_dim}D input, got {z.size(-1)}D"
        )
        x = z

        for i, conv in enumerate(self.decoder_layers):
            if i < len(self.decoder_layers) - 1:
                bn = self.decoder_bns[i] if self.batch_norm else None
                x = conv_forward(
                    conv,
                    x,
                    edge_index,
                    edge_attr,
                    bn=bn,
                    batch=batch,
                    dropout_p=self.dropout_rate,
                    training=self.training,
                    use_checkpointing=self.use_checkpointing,
                )
            else:  # Last decoder layer — sigmoid constrains output to [0,1]
                x = torch.sigmoid(
                    conv_forward(
                        conv,
                        x,
                        edge_index,
                        edge_attr,
                        activation=None,
                        use_checkpointing=self.use_checkpointing,
                    )
                )
        cont_out = x  # shape: [num_nodes, in_channels]
        canid_logits = self.canid_classifier(z)

        return cont_out, canid_logits

    def create_neighborhood_targets(self, node_id, edge_index, batch):
        """Create neighborhood target matrix for training.

        Args:
            node_id: Global CAN ID indices [num_nodes].
            edge_index: Edge indices [2, num_edges].
            batch: Batch assignment vector.

        Returns:
            Binary target matrix [num_nodes, num_ids].
        """
        num_nodes = node_id.size(0)
        neighbor_targets = torch.zeros(num_nodes, self.num_ids, device=node_id.device)

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        dst_can_ids = node_id[dst_nodes]

        valid = (dst_can_ids >= 0) & (dst_can_ids < self.num_ids)
        neighbor_targets[src_nodes[valid], dst_can_ids[valid]] = 1.0

        return neighbor_targets

    @staticmethod
    def neighborhood_loss_negsampled(
        logits: torch.Tensor,
        node_id: torch.Tensor,
        edge_index: torch.Tensor,
        num_ids: int,
        k_neg: int = 32,
    ) -> torch.Tensor:
        """Neighborhood BCE loss with negative sampling.

        Memory: O(num_edges + num_nodes * k_neg) instead of O(num_nodes * num_ids).
        """
        src, dst = edge_index
        dst_ids = node_id[dst]
        valid = (dst_ids >= 0) & (dst_ids < num_ids)
        # Positive: logits at true neighbor IDs
        pos_logits = logits[src[valid], dst_ids[valid]]
        pos_loss = -F.logsigmoid(pos_logits).mean()
        # Negative: random IDs per node, excluding true neighbors (sparse rejection)
        neg_ids = torch.randint(0, num_ids, (logits.size(0), k_neg), device=logits.device)
        # Encode (node, id) as unique keys for sparse collision check
        pos_keys = src[valid].long() * num_ids + dst_ids[valid].long()
        node_range = torch.arange(logits.size(0), device=logits.device)
        neg_keys = (node_range.unsqueeze(1) * num_ids + neg_ids).reshape(-1)
        is_collision = torch.isin(neg_keys, pos_keys)
        neg_logits = logits.gather(1, neg_ids).reshape(-1)[~is_collision]
        neg_loss = (
            -F.logsigmoid(-neg_logits).mean() if neg_logits.numel() > 0 else logits.new_zeros(1)
        )
        return pos_loss + neg_loss

    @classmethod
    def from_config(cls, cfg, num_ids: int, in_ch: int) -> GraphAutoencoderNeighborhood:
        """Construct from a config."""
        conv_type = cfg.conv_type
        return cls(
            num_ids=num_ids,
            in_channels=in_ch,
            hidden_dims=list(cfg.hidden_dims),
            latent_dim=cfg.latent_dim,
            encoder_heads=cfg.heads,
            embedding_dim=cfg.embedding_dim,
            dropout=cfg.dropout,
            conv_type=conv_type,
            edge_dim=resolve_edge_dim(conv_type, cfg.edge_dim),
            proj_dim=cfg.proj_dim,
            use_checkpointing=cfg.gradient_checkpointing,
            variational=getattr(cfg, "variational", True),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, mask_ratio: float = 0.0, node_id=None):
        """Forward pass through the GraphAutoencoderNeighborhood.

        Args:
            x: Continuous node features [num_nodes, in_channels].
            edge_index: Edge indices [2, num_edges].
            batch: Batch assignment vector.
            edge_attr: Optional edge features for TransformerConv/GATv2Conv.
            mask_ratio: Fraction of features to mask during training
                (GraphMAE-style). Masked features are zeroed before encoding;
                the returned mask indicates which (node, feature) positions
                were masked for selective reconstruction loss. Set to 0.0 to
                disable (inference, or legacy behavior).
            node_id: Global CAN ID indices [num_nodes] for embedding lookup.

        Returns:
            tuple: (cont_out, canid_logits, neighbor_logits, z, kl_loss, mask).
            mask is a bool tensor [num_nodes, in_channels] or None if mask_ratio=0.
        """
        mask = None
        if mask_ratio > 0.0 and self.training:
            mask = torch.rand_like(x) < mask_ratio
            x = x.clone()
            x[mask] = 0.0

        ea = edge_attr if self._uses_edge_attr else None
        z, kl_loss = self.encode(x, edge_index, edge_attr=ea, batch=batch, node_id=node_id)
        cont_out, canid_logits = self.decode_node(z, edge_index, edge_attr=ea, batch=batch)
        neighbor_logits = self.neighborhood_decoder(z)
        return cont_out, canid_logits, neighbor_logits, z, kl_loss, mask

    @torch.no_grad()
    def score_difficulty(
        self,
        graphs: list,
        canid_weight: float = 1.0,
        batch_size: int = 500,
    ) -> list[float]:
        """Score reconstruction difficulty for curriculum learning.

        Per-graph score = mean_node_MSE + canid_weight * mean_node_CE.
        Higher score = harder to reconstruct = more difficult sample.
        """
        from torch_geometric.loader import DataLoader as PyGDataLoader
        from torch_geometric.utils import scatter

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            scores: list[float] = []
            for batch in PyGDataLoader(graphs, batch_size=batch_size):
                batch = batch.clone().to(device, non_blocking=True)
                edge_attr = getattr(batch, "edge_attr", None)
                cont, canid_logits, _, _, _, _ = self(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    edge_attr=edge_attr,
                    node_id=batch.node_id,
                )
                node_mse = (cont - batch.x).pow(2).mean(dim=1)
                graph_mse = scatter(node_mse, batch.batch, reduce="mean")
                node_ce = F.cross_entropy(canid_logits, batch.node_id, reduction="none")
                graph_ce = scatter(node_ce, batch.batch, reduce="mean")
                scores.extend((graph_mse + canid_weight * graph_ce).tolist())
            return scores
        finally:
            self.train(was_training)
