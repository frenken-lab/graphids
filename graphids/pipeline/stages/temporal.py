"""Temporal graph classification stage.

Loads a pretrained GAT spatial encoder, wraps it with a Transformer-based
temporal head, and trains on sequences of consecutive graph snapshots.

Uses contiguous time split (not random): first 80% train, last 20% val
because temporal ordering matters for this task.
"""

from __future__ import annotations

import structlog
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)


from graphids.core.preprocessing import CANBusDataModule

from .data_loading import cleanup
from .trainer_factory import load_model, make_trainer

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Dataset for temporal sequences
# ---------------------------------------------------------------------------


class TemporalGraphDataset(Dataset):
    """PyTorch Dataset wrapping a list of GraphSequence objects."""

    def __init__(self, sequences, device):
        self.sequences = sequences
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq.graphs, seq.y


def collate_temporal(batch):
    """Custom collate for temporal graph sequences.

    Returns:
        graph_sequences: list of lists of Data objects
        labels: tensor of labels
    """
    graph_sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return graph_sequences, labels


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class TemporalLightningModule(pl.LightningModule):
    """Lightning wrapper for TemporalGraphClassifier."""

    def __init__(self, model, cfg):
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

    def _shared_step(self, batch, stage: str):
        graph_sequences, labels = batch
        device = self.device

        # Move graphs to device
        moved_sequences = []
        for seq in graph_sequences:
            moved_sequences.append([g.clone().to(device) for g in seq])

        logits = self.model(moved_sequences)
        loss = F.cross_entropy(logits, labels.to(device))

        preds = logits.argmax(dim=1)
        acc = (preds == labels.to(device)).float().mean()

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
        moved_sequences = [[g.clone().to(device) for g in seq] for seq in graph_sequences]
        logits = self.model(moved_sequences)
        preds = logits.argmax(dim=1)
        self.test_metrics.update(preds, labels.to(device))

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        t = self.cfg.training
        tc = self.cfg.temporal

        # Separate param groups for spatial encoder (low LR) and temporal head (full LR)
        spatial_params = list(self.model.spatial_encoder.parameters())
        temporal_params = [
            p
            for n, p in self.model.named_parameters()
            if not n.startswith("spatial_encoder") and p.requires_grad
        ]

        param_groups = []
        if not tc.freeze_spatial and spatial_params:
            param_groups.append(
                {
                    "params": spatial_params,
                    "lr": t.lr * tc.spatial_lr_factor,
                }
            )
        if temporal_params:
            param_groups.append(
                {
                    "params": temporal_params,
                    "lr": t.lr,
                }
            )

        optimizer = torch.optim.AdamW(
            param_groups if param_groups else self.model.parameters(),
            lr=t.lr,
            weight_decay=t.weight_decay,
        )
        return optimizer


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_temporal(cfg) -> dict:
    """Train temporal graph classifier.

    Loads pretrained GAT, wraps with temporal Transformer head, trains
    on sequences of consecutive graph snapshots.
    """
    tc = cfg.temporal
    if not tc.enabled:
        log.warning("Temporal training called but temporal.enabled=False. Skipping.")
        return {"status": "skipped", "reason": "temporal.enabled=False"}

    log.info(
        "temporal_stage_start",
        dataset=cfg.dataset,
        model_type=cfg.model_type,
        scale=cfg.scale,
        window=tc.temporal_window,
        stride=tc.temporal_stride,
    )
    pl.seed_everything(cfg.seed)

    # Load data
    dm = CANBusDataModule.from_cfg(cfg)
    dm.setup("fit")
    dm.populate_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load pretrained GAT
    gat_stage = "curriculum"
    gat = load_model(cfg, "gat", gat_stage, device)
    log.info("loaded_pretrained_gat", stage=gat_stage)

    # Probe spatial embedding dim
    with torch.no_grad():
        probe = dm.train_dataset[0].clone().to(device)
        _, emb = gat(probe, return_embedding=True)
        spatial_dim = emb.shape[-1]
    log.info("spatial_embedding_dim", dim=spatial_dim)

    # Group into temporal sequences
    from graphids.core.preprocessing._temporal import TemporalGrouper

    grouper = TemporalGrouper(window=tc.temporal_window, stride=tc.temporal_stride)

    # Contiguous time split: first 80% train, last 20% val
    all_graphs = list(dm.train_dataset) + list(dm.val_dataset)
    split_idx = int(len(all_graphs) * tc.train_split)
    temporal_train_graphs = all_graphs[:split_idx]
    temporal_val_graphs = all_graphs[split_idx:]

    train_sequences = grouper.group(temporal_train_graphs)
    val_sequences = grouper.group(temporal_val_graphs)

    log.info(
        "temporal_sequences",
        train=len(train_sequences),
        val=len(val_sequences),
        total_graphs=len(all_graphs),
    )

    if not train_sequences or not val_sequences:
        log.error("Not enough graphs for temporal windowing")
        return {"status": "failed", "reason": "insufficient graphs for windowing"}

    # Create dataloaders
    train_ds = TemporalGraphDataset(train_sequences, device)
    val_ds = TemporalGraphDataset(val_sequences, device)

    # Small batch size for temporal (sequences are heavy)
    temporal_batch_size = max(1, min(32, len(train_sequences) // 10))

    train_loader = DataLoader(
        train_ds,
        batch_size=temporal_batch_size,
        shuffle=True,
        collate_fn=collate_temporal,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=temporal_batch_size,
        shuffle=False,
        collate_fn=collate_temporal,
        num_workers=0,
    )

    # Build temporal model
    from graphids.core.models.temporal import TemporalGraphClassifier

    temporal_model = TemporalGraphClassifier(
        spatial_encoder=gat,
        spatial_dim=spatial_dim,
        temporal_hidden=tc.temporal_hidden,
        temporal_heads=tc.temporal_heads,
        temporal_layers=tc.temporal_layers,
        max_seq_len=tc.temporal_window,
        freeze_spatial=tc.freeze_spatial,
        num_classes=cfg.num_classes,
    ).to(device)

    total_params = sum(p.numel() for p in temporal_model.parameters())
    trainable_params = sum(p.numel() for p in temporal_model.parameters() if p.requires_grad)
    log.info("temporal_model_params", total=total_params, trainable=trainable_params)

    # run_stage() already saves config.yaml

    # Train
    lit_module = TemporalLightningModule(temporal_model, cfg)
    trainer = make_trainer(cfg, "temporal")
    trainer.fit(lit_module, train_loader, val_loader)

    # Restore best-epoch weights before saving
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        best_ckpt = torch.load(best_path, weights_only=True)
        lit_module.load_state_dict(best_ckpt["state_dict"])
    torch.save(temporal_model.state_dict(), "best_model.pt")
    log.info("saved_temporal_model", checkpoint="best_model.pt")

    # Compute final metrics via Lightning test loop
    from .eval_inference import extract_metrics, make_test_trainer

    test_trainer = make_test_trainer()
    test_trainer.test(lit_module, dataloaders=val_loader, verbose=False)
    metrics = extract_metrics(lit_module)
    metrics["core"]["n_sequences"] = len(val_sequences)
    metrics["core"]["window"] = tc.temporal_window
    metrics["core"]["stride"] = tc.temporal_stride
    result_metrics = {"temporal": metrics}

    log.info(
        "temporal_metrics",
        **{k: round(v, 4) for k, v in result_metrics["temporal"]["core"].items() if isinstance(v, float)},
    )

    cleanup()
    return {"metrics": result_metrics}
