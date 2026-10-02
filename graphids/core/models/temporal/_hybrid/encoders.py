"""Input and time encoders for the temporal hybrid model."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalInputEncoder(nn.Module):
    """Encode event bytes/features plus source and destination ID embeddings."""

    def __init__(
        self,
        *,
        num_ids: int,
        in_channels: int,
        embedding_dim: int,
        hidden: int,
        dropout: float,
        use_id_embeddings: bool = True,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(max(1, int(num_ids)), int(embedding_dim)) if use_id_embeddings else None
        self.dst_embedding = nn.Embedding(max(1, int(num_ids)), int(embedding_dim)) if use_id_embeddings else None
        input_dim = int(in_channels) + (2 * int(embedding_dim) if use_id_embeddings else 0)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, batch) -> torch.Tensor:
        x = batch.msg.float()
        if self.src_embedding is not None and self.dst_embedding is not None:
            src = batch.src.clamp_min(0).clamp_max(self.src_embedding.num_embeddings - 1).long()
            dst = batch.dst.clamp_min(0).clamp_max(self.dst_embedding.num_embeddings - 1).long()
            x = torch.cat([x, self.src_embedding(src), self.dst_embedding(dst)], dim=-1)
        return self.proj(x)


class TemporalTimeEncoder(nn.Module):
    """Sinusoidal encoding for positive elapsed times."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = max(1, int(dim))
        half = max(1, (self.dim + 1) // 2)
        self.register_buffer("freqs", torch.logspace(0, 3, steps=half), persistent=False)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(delta.float().clamp_min(0)).unsqueeze(-1) / self.freqs
        encoded = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return encoded[..., : self.dim]
