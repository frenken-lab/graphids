"""Temporal Graph Classifier: spatial GNN encoder + Transformer over time.

Architecture:
  1. Shared GATWithJK extracts per-snapshot graph embedding
     (optionally frozen or with reduced LR).
  2. nn.TransformerEncoder processes the sequence [batch, time, spatial_dim].
  3. FC head classifies from the last timestep's hidden state.

No new dependencies — uses PyTorch native nn.TransformerEncoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalGraphClassifier(nn.Module):
    """Temporal attention over spatial graph embeddings.

    Args:
        spatial_encoder: Pretrained GATWithJK (or similar) that produces
            graph-level embeddings via forward(data, return_embedding=True).
        spatial_dim: Dimensionality of the spatial encoder's embedding output.
        temporal_hidden: Hidden dimension for the Transformer.
        temporal_heads: Number of attention heads.
        temporal_layers: Number of TransformerEncoder layers.
        max_seq_len: Maximum sequence length for positional encoding.
        freeze_spatial: If True, spatial encoder weights are frozen.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        spatial_encoder: nn.Module,
        spatial_dim: int,
        temporal_hidden: int = 64,
        temporal_heads: int = 4,
        temporal_layers: int = 2,
        max_seq_len: int = 32,
        freeze_spatial: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()
        self.spatial_encoder = spatial_encoder
        self.freeze_spatial = freeze_spatial

        if freeze_spatial:
            for p in self.spatial_encoder.parameters():
                p.requires_grad = False
            self.spatial_encoder.eval()

        # Project spatial embedding to temporal hidden dim
        self.spatial_proj = nn.Linear(spatial_dim, temporal_hidden)

        # Learned positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, temporal_hidden)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_hidden,
            nhead=temporal_heads,
            dim_feedforward=temporal_hidden * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_hidden),
            nn.Linear(temporal_hidden, temporal_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(temporal_hidden, num_classes),
        )

    def forward(self, graph_sequences: list[list]) -> torch.Tensor:
        """Forward pass on a batch of graph sequences.

        Args:
            graph_sequences: List of lists of PyG Data objects.
                Outer list = batch, inner list = time steps.

        Returns:
            Logits of shape [batch_size, num_classes].
        """
        batch_size = len(graph_sequences)
        seq_len = len(graph_sequences[0])
        device = next(self.parameters()).device

        # Encode all graphs spatially
        all_embeddings = []
        for seq in graph_sequences:
            seq_embs = []
            ctx = torch.no_grad() if self.freeze_spatial else torch.enable_grad()
            with ctx:
                if self.freeze_spatial:
                    self.spatial_encoder.eval()
                for g in seq:
                    _, emb = self.spatial_encoder(g, return_embedding=True)
                    seq_embs.append(emb.squeeze(0))  # [spatial_dim]
            all_embeddings.append(torch.stack(seq_embs))  # [seq_len, spatial_dim]

        # [batch_size, seq_len, spatial_dim]
        x = torch.stack(all_embeddings)

        # Project to temporal hidden dim
        x = self.spatial_proj(x)  # [batch, seq_len, temporal_hidden]

        # Add positional encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)

        # Transformer encoder
        x = self.transformer(x)  # [batch, seq_len, temporal_hidden]

        # Classify from last timestep
        x = x[:, -1, :]  # [batch, temporal_hidden]
        logits = self.classifier(x)  # [batch, num_classes]

        return logits
