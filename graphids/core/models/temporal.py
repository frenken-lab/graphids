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


def _probe_spatial_dim(gat: nn.Module, cfg) -> int:
    """Derive spatial embedding dim from the GAT architecture without data.

    Uses the JK output dim (hidden_channels * heads) scaled by the number
    of pool aggregators, matching what ``GATWithJK.forward(return_embedding=True)``
    produces after global pooling.
    """
    jk_dim = cfg.gat.hidden * cfg.gat.heads
    n_aggrs = len(cfg.gat.pool_aggrs) if hasattr(cfg.gat, "pool_aggrs") else 1
    return jk_dim * n_aggrs


class TemporalLightningModule(pl.LightningModule):
    """Lightning wrapper for TemporalGraphClassifier.

    Constructor builds the model internally from config so Lightning's
    ``save_hyperparameters`` / ``load_from_checkpoint`` round-trip works.

    Args:
        cfg: Config namespace (or plain dict on reload from checkpoint).
        gat_ckpt_path: Path to pretrained GAT checkpoint. Required for
            training; None during ``load_from_checkpoint`` reconstruction
            (weights come from the Lightning checkpoint itself).
    """

    def __init__(self, cfg, gat_ckpt_path: str | None = None):
        super().__init__()
        from graphids.config import to_namespace
        cfg = to_namespace(cfg)

        # cfg as plain dict; gat_ckpt_path not needed on reload (weights come
        # from the Lightning checkpoint itself).
        self.save_hyperparameters({"cfg": cfg.as_dict()}, ignore=["gat_ckpt_path"])

        self.cfg = cfg
        self.model = self._build_model(cfg, gat_ckpt_path)
        self.test_metrics = MetricCollection({
            "accuracy": BinaryAccuracy(), "f1": BinaryF1Score(),
            "precision": BinaryPrecision(), "recall": BinaryRecall(),
            "specificity": BinarySpecificity(), "auc": BinaryAUROC(),
        })

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model(cfg, gat_ckpt_path: str | None) -> TemporalGraphClassifier:
        """Build TemporalGraphClassifier from config.

        When *gat_ckpt_path* is provided (training), loads the pretrained GAT
        and probes its embedding dim.  When None (``load_from_checkpoint``),
        builds a skeleton GAT from config dimensions — the real weights will be
        loaded from the Lightning checkpoint's ``state_dict``.
        """
        from pathlib import Path

        from graphids.core.models.gat import GATWithJK

        tc = cfg.temporal

        if gat_ckpt_path is not None:
            # Training path: load pretrained GAT from checkpoint
            ckpt_path = Path(gat_ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"GAT checkpoint not found: {ckpt_path}\n"
                    f"The GAT stage must be trained first."
                )
            gat = GATWithJK.from_config(cfg, cfg.num_ids, cfg.in_channels)
            checkpoint = torch.load(gat_ckpt_path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                raw = checkpoint["state_dict"]
                checkpoint = (
                    {k.replace("model.", ""): v for k, v in raw.items() if k.startswith("model.")}
                    or raw
                )
            gat.load_state_dict(checkpoint)
            gat.eval()

            # Probe spatial embedding dim
            spatial_dim = _probe_spatial_dim(gat, cfg)
        else:
            # Reconstruction path (load_from_checkpoint): skeleton GAT,
            # weights will be overwritten by Lightning's state_dict restore.
            gat = GATWithJK.from_config(cfg, cfg.num_ids, cfg.in_channels)
            spatial_dim = _probe_spatial_dim(gat, cfg)

        model = TemporalGraphClassifier(
            spatial_encoder=gat,
            spatial_dim=spatial_dim,
            temporal_hidden=tc.temporal_hidden,
            temporal_heads=tc.temporal_heads,
            temporal_layers=tc.temporal_layers,
            max_seq_len=tc.temporal_window,
            freeze_spatial=tc.freeze_spatial,
            num_classes=cfg.num_classes,
        )
        return model

    @classmethod
    def from_datamodule(cls, cfg, dm) -> TemporalLightningModule:
        """Build from config + TemporalDataModule.

        Extracts the GAT checkpoint path from cfg; the DataModule is only
        used for device placement after construction.
        """
        import structlog

        gat_ckpt_path = str(cfg.checkpoints["gat"])
        module = cls(cfg, gat_ckpt_path=gat_ckpt_path)
        module.model = module.model.to(dm.device)
        dm.gat = None  # DataModule no longer needs its GAT reference

        total_params = sum(p.numel() for p in module.model.parameters())
        trainable = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
        structlog.get_logger().info("temporal_model_params", total=total_params, trainable=trainable)

        return module

    def forward(self, graph_sequences):
        return self.model(graph_sequences)

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

    @classmethod
    def evaluate(cls, cfg, val_data, test_scenarios, device, *, load_model_fn) -> dict | None:
        """Evaluate temporal model via Lightning test loop.

        Loads the full Lightning checkpoint via ``load_from_checkpoint``,
        avoiding manual model reconstruction.
        """
        from torch.utils.data import DataLoader

        from graphids.core.preprocessing._temporal import (
            TemporalGraphDataset,
            TemporalGrouper,
            collate_temporal,
        )

        from ._training import gpu_cleanup, test_model

        ckpt_path = cfg.checkpoints["temporal"]
        module = cls.load_from_checkpoint(ckpt_path, map_location=device, weights_only=True)
        module = module.to(device)
        module.eval()

        tc = cfg.temporal
        grouper = TemporalGrouper(window=tc.temporal_window, stride=tc.temporal_stride)
        val_sequences = grouper.group(val_data)
        if not val_sequences:
            return None

        val_loader = DataLoader(
            TemporalGraphDataset(val_sequences, device),
            batch_size=32, shuffle=False,
            collate_fn=collate_temporal, num_workers=0,
        )
        val_metrics = test_model(module, val_loader)

        gpu_cleanup(module.model)
        return {"val_metrics": val_metrics, "test_metrics": {}, "artifacts": None}
