import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv

from ._utils import checkpoint_conv


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
    ):
        super().__init__()
        # ID embedding: expect real torch.nn.Embedding to be available in test env
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        self.num_ids = num_ids
        self.dropout_rate = dropout
        self.batch_norm = batch_norm
        self.use_checkpointing = use_checkpointing
        self.conv_type = conv_type
        self._uses_edge_attr = conv_type in ("transformer", "gatv2")
        self._edge_dim = edge_dim if self._uses_edge_attr else None
        self._proj_dim = proj_dim

        # Optional input projection: decouple feature count from architecture
        if proj_dim > 0:
            self.feat_proj = nn.Linear(in_channels - 1, proj_dim)
        else:
            self.feat_proj = None

        # Hidden dims schedule: interpret list; if last equals latent_dim assume the
        # list includes latent entry and use hidden_dims[:-1] as encoder targets.
        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = [max(128, latent_dim * 2), latent_dim]

        # If last hidden dim equals latent_dim, drop it for encoder targets
        if len(hidden_dims) >= 2 and hidden_dims[-1] == latent_dim:
            encoder_targets = hidden_dims[:-1]
        else:
            encoder_targets = hidden_dims

        # Input dim to first GAT combines ID embedding and continuous features
        cont_dim = proj_dim if proj_dim > 0 else (in_channels - 1)
        gat_in_dim = embedding_dim + cont_dim
        self.gat_in_dim = gat_in_dim

        # Encoder: build progressive GAT layers matching encoder_targets
        self.encoder_layers = nn.ModuleList()
        self.encoder_bns = nn.ModuleList()
        in_dim = gat_in_dim
        for i, target_dim in enumerate(encoder_targets):
            heads = encoder_heads if i == 0 else 1
            # per-head out dim
            if heads > 1 and target_dim % heads == 0:
                out_per_head = target_dim // heads
            else:
                heads = 1
                out_per_head = target_dim
            self.encoder_layers.append(
                _make_conv(
                    conv_type,
                    in_dim,
                    out_per_head,
                    heads=heads,
                    edge_dim=self._edge_dim,
                )
            )
            if self.batch_norm:
                self.encoder_bns.append(nn.BatchNorm1d(target_dim))
            in_dim = target_dim

        # Latent heads map final encoder output to latent_dim
        self.latent_in_dim = in_dim
        self.z_mean = nn.Linear(self.latent_in_dim, latent_dim)
        self.z_logvar = nn.Linear(self.latent_in_dim, latent_dim)

        # Decoder: mirror of encoder (reverse progressive schedule)
        decoder_targets = list(reversed(encoder_targets))
        self.decoder_layers = nn.ModuleList()
        self.decoder_bns = nn.ModuleList()
        in_dim = latent_dim

        # Validate: first decoder layer must accept latent_dim input
        if len(decoder_targets) == 0:
            raise ValueError(
                f"decoder_targets is empty! hidden_dims={hidden_dims}, encoder_targets={encoder_targets}"
            )
        for i, target_dim in enumerate(decoder_targets):
            # For intermediate decoder layers we may use multiple heads; final layer maps to continuous features
            is_last = i == len(decoder_targets) - 1
            heads = decoder_heads if (not is_last and decoder_heads > 1) else 1
            if heads > 1 and target_dim % heads == 0 and not is_last:
                out_per_head = target_dim // heads
            else:
                heads = 1
                out_per_head = target_dim if not is_last else (in_channels - 1)

            self.decoder_layers.append(
                _make_conv(
                    conv_type,
                    in_dim,
                    out_per_head,
                    heads=heads,
                    edge_dim=self._edge_dim,
                )
            )
            if (not is_last) and self.batch_norm:
                self.decoder_bns.append(nn.BatchNorm1d(out_per_head * heads))
            # next in_dim for following layer
            in_dim = (out_per_head * heads) if (not is_last) else (in_channels - 1)

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

    def encode(self, x, edge_index, edge_attr=None):
        # Use the embedding's forward interface (works for real Embedding and SimpleEmbedding)
        id_emb = self.id_embedding(x[:, 0].long())
        other_feats = x[:, 1:]
        if self.feat_proj is not None:
            other_feats = self.feat_proj(other_feats)
        x = torch.cat([id_emb, other_feats], dim=1)
        # Apply encoder layers; handle optional batchnorm safely
        for i, conv in enumerate(self.encoder_layers):
            if self.use_checkpointing and x.requires_grad:
                x = checkpoint_conv(conv, x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index, edge_attr) if edge_attr is not None else conv(x, edge_index)
            if self.batch_norm:
                bn = self.encoder_bns[i]
                x = self.dropout(F.relu(bn(x)))
            else:
                x = self.dropout(F.relu(x))
        mu = self.z_mean(x)
        logvar = self.z_logvar(x).clamp(-20, 20)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl_loss

    def decode_node(self, z, edge_index, edge_attr=None):
        x = z

        # Runtime shape validation
        if x.size(-1) != self.latent_dim:
            raise RuntimeError(
                f"decode_node input has {x.size(-1)} features but expected latent_dim={self.latent_dim}"
            )

        # Check first decoder layer expects latent_dim input
        first_layer = self.decoder_layers[0]
        if first_layer.in_channels != self.latent_dim:
            raise RuntimeError(
                f"First decoder layer expects {first_layer.in_channels} features "
                f"but latent_dim={self.latent_dim}. "
                f"Decoder layers: {[(l.in_channels, l.out_channels, l.heads) for l in self.decoder_layers]}"
            )

        for i, conv in enumerate(self.decoder_layers):
            if i < len(self.decoder_layers) - 1:
                if self.use_checkpointing and x.requires_grad:
                    x = checkpoint_conv(conv, x, edge_index, edge_attr)
                else:
                    x = (
                        conv(x, edge_index, edge_attr)
                        if edge_attr is not None
                        else conv(x, edge_index)
                    )
                if self.batch_norm:
                    bn = self.decoder_bns[i]
                    x = self.dropout(F.relu(bn(x)))
                else:
                    x = self.dropout(F.relu(x))
            else:
                if self.use_checkpointing and x.requires_grad:
                    x = torch.sigmoid(checkpoint_conv(conv, x, edge_index, edge_attr))
                else:
                    x = torch.sigmoid(
                        conv(x, edge_index, edge_attr)
                        if edge_attr is not None
                        else conv(x, edge_index)
                    )
        cont_out = x  # shape: [num_nodes, in_channels-1]
        canid_logits = self.canid_classifier(z)

        return cont_out, canid_logits

    def decode_neighborhood(self, z):
        """Decode latent representation to neighborhood predictions.

        Args:
            z (torch.Tensor): Latent node embeddings with shape [num_nodes, latent_dim].

        Returns:
            torch.Tensor: Neighborhood logits with shape [num_nodes, num_ids].
        """
        neighbor_logits = self.neighborhood_decoder(z)
        return neighbor_logits

    def create_neighborhood_targets(self, x, edge_index, batch):
        """Create neighborhood target matrix for training.

        Args:
            x (torch.Tensor): Node features with CAN IDs in first column.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor): Batch assignment vector.

        Returns:
            torch.Tensor: Binary target matrix [num_nodes, num_ids].
        """
        num_nodes = x.size(0)
        device = x.device
        neighbor_targets = torch.zeros(num_nodes, self.num_ids, device=device)

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        dst_can_ids = x[dst_nodes, 0].long()

        valid = (dst_can_ids >= 0) & (dst_can_ids < self.num_ids)
        src_valid = src_nodes[valid]
        dst_ids_valid = dst_can_ids[valid]

        neighbor_targets[src_valid, dst_ids_valid] = 1.0

        return neighbor_targets

    @classmethod
    def from_config(cls, cfg, num_ids: int, in_ch: int) -> "GraphAutoencoderNeighborhood":
        """Construct from a PipelineConfig."""
        conv_type = cfg.vgae.conv_type
        return cls(
            num_ids=num_ids,
            in_channels=in_ch,
            hidden_dims=list(cfg.vgae.hidden_dims),
            latent_dim=cfg.vgae.latent_dim,
            encoder_heads=cfg.vgae.heads,
            embedding_dim=cfg.vgae.embedding_dim,
            dropout=cfg.vgae.dropout,
            conv_type=conv_type,
            edge_dim=cfg.vgae.edge_dim if conv_type in ("transformer", "gatv2") else None,
            proj_dim=cfg.vgae.proj_dim,
            use_checkpointing=cfg.training.gradient_checkpointing,
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        """Forward pass through the GraphAutoencoderNeighborhood.

        Args:
            x (torch.Tensor): Node features with shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Edge indices with shape [2, num_edges].
            batch (torch.Tensor): Batch assignment vector.
            edge_attr (torch.Tensor, optional): Edge features for TransformerConv/GATv2Conv.

        Returns:
            tuple: (continuous_output, canid_logits, neighbor_logits, latent_embeddings, kl_loss).
        """
        ea = edge_attr if self._uses_edge_attr else None
        z, kl_loss = self.encode(x, edge_index, edge_attr=ea)
        cont_out, canid_logits = self.decode_node(z, edge_index, edge_attr=ea)
        neighbor_logits = self.decode_neighborhood(z)
        return cont_out, canid_logits, neighbor_logits, z, kl_loss
