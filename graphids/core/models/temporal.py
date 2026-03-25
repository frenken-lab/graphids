"""Temporal Graph Classifier: spatial GNN encoder + Transformer over time.

Architecture:
  1. Shared GATWithJK extracts per-snapshot graph embedding
     (optionally frozen or with reduced LR).
  2. nn.TransformerEncoder processes the sequence [batch, time, spatial_dim].
  3. FC head classifies from the last timestep's hidden state.

No new dependencies — uses PyTorch native nn.TransformerEncoder.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)


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

        # Encode all graphs spatially in one batched forward pass
        from torch_geometric.data import Batch

        assert all(len(seq) == seq_len for seq in graph_sequences), (
            "All sequences must have equal length for batched spatial encoding"
        )
        all_graphs = [g for seq in graph_sequences for g in seq]
        big_batch = Batch.from_data_list(all_graphs).to(device)

        ctx = torch.no_grad() if self.freeze_spatial else torch.enable_grad()
        with ctx:
            if self.freeze_spatial:
                self.spatial_encoder.eval()
            _, all_embs = self.spatial_encoder(big_batch, return_embedding=True)

        # all_embs is [total_graphs, spatial_dim] — reshape to [batch, seq_len, spatial_dim]
        x = all_embs.view(batch_size, seq_len, -1)

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


class TemporalLightningModule(pl.LightningModule):
    """Lightning wrapper for TemporalGraphClassifier."""

    def __init__(self, model: TemporalGraphClassifier, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.test_metrics = MetricCollection({
            "accuracy": BinaryAccuracy(), "f1": BinaryF1Score(),
            "precision": BinaryPrecision(), "recall": BinaryRecall(),
            "specificity": BinarySpecificity(), "auc": BinaryAUROC(),
        })

    def forward(self, graph_sequences):
        return self.model(graph_sequences)

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def _shared_step(self, batch, stage: str):
        graph_sequences, labels = batch
        device = self.device

        moved_sequences = []
        for seq in graph_sequences:
            moved_sequences.append([g.clone().to(device, non_blocking=True) for g in seq])

        logits = self.model(moved_sequences)
        loss = F.cross_entropy(logits, labels.to(device, non_blocking=True))

        preds = logits.argmax(dim=1)
        acc = (preds == labels.to(device, non_blocking=True)).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=len(graph_sequences))
        self.log(f"{stage}_acc", acc, prog_bar=True, batch_size=len(graph_sequences))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        graph_sequences, labels = batch
        device = self.device
        moved_sequences = [[g.clone().to(device, non_blocking=True) for g in seq] for seq in graph_sequences]
        logits = self.model(moved_sequences)
        scores = F.softmax(logits, dim=1)[:, 1]
        self.test_metrics.update(scores, labels.to(device, non_blocking=True))

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        t = self.cfg.training
        tc = self.cfg.temporal

        spatial_params = list(self.model.spatial_encoder.parameters())
        temporal_params = [
            p
            for n, p in self.model.named_parameters()
            if not n.startswith("spatial_encoder") and p.requires_grad
        ]

        param_groups = []
        if not tc.freeze_spatial and spatial_params:
            param_groups.append({"params": spatial_params, "lr": t.lr * tc.spatial_lr_factor})
        if temporal_params:
            param_groups.append({"params": temporal_params, "lr": t.lr})

        return torch.optim.AdamW(
            param_groups if param_groups else self.model.parameters(),
            lr=t.lr, weight_decay=t.weight_decay,
        )
