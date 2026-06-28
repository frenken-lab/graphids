"""Temporal supervised attention model for CAN event streams."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphids.core.losses import CrossEntropyLoss
from graphids.core.models.base import classification_test_metrics

from .base import TemporalModuleBase


class TemporalGAT(TemporalModuleBase):
    """Causal event-attention classifier over PyG ``TemporalData`` batches."""

    _SCALES: dict[str, dict[str, int]] = {
        "small": {"hidden": 64, "layers": 2, "heads": 4, "embedding_dim": 16},
        "large": {"hidden": 128, "layers": 3, "heads": 8, "embedding_dim": 32},
    }

    def __init__(
        self,
        *,
        loss_fn: nn.Module | None = None,
        hidden: int | None = None,
        layers: int | None = None,
        heads: int | None = None,
        embedding_dim: int | None = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scale: str = "small",
        model_type: str = "temporal_gat",
        dataset: str = "",
        seed: int = 42,
        num_ids: int = 0,
        in_channels: int = 0,
        num_classes: int = 2,
    ):
        loss_fn = loss_fn if loss_fn is not None else CrossEntropyLoss()
        preset = self._SCALES.get(scale, {})
        hidden = hidden if hidden is not None else preset.get("hidden", 64)
        layers = layers if layers is not None else preset.get("layers", 2)
        heads = heads if heads is not None else preset.get("heads", 4)
        embedding_dim = embedding_dim if embedding_dim is not None else preset.get("embedding_dim", 16)
        super().__init__()
        self.test_metrics = classification_test_metrics(num_classes)
        self._val_probs: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._init_post(locals())

    def _build(self) -> None:
        hp = self.hparams
        hidden = int(hp.hidden)
        heads = int(hp.heads)
        if hidden % heads != 0:
            raise ValueError("temporal_gat hidden size must be divisible by heads")
        self.src_embedding = nn.Embedding(max(1, int(hp.num_ids)), int(hp.embedding_dim))
        self.dst_embedding = nn.Embedding(max(1, int(hp.num_ids)), int(hp.embedding_dim))
        input_dim = int(hp.in_channels) + (2 * int(hp.embedding_dim))
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(float(hp.dropout)),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=float(hp.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(hp.layers))
        self.head = nn.Linear(hidden, int(hp.num_classes))
        self.test_metrics = classification_test_metrics(int(hp.num_classes))

    @staticmethod
    def _rebuild_excluded_kwargs(hp: dict) -> dict:
        from graphids.core.losses.build import build_loss

        return {"loss_fn": build_loss("temporal_gat", hp.get("loss_config"))}

    def _event_input(self, batch) -> torch.Tensor:
        src = batch.src.clamp_min(0).clamp_max(self.src_embedding.num_embeddings - 1).long()
        dst = batch.dst.clamp_min(0).clamp_max(self.dst_embedding.num_embeddings - 1).long()
        return torch.cat([batch.msg.float(), self.src_embedding(src), self.dst_embedding(dst)], dim=-1)

    def forward_temporal(self, batch, state=None) -> torch.Tensor:
        del state
        x = self.input_proj(self._event_input(batch)).unsqueeze(0)
        n_events = x.size(1)
        causal_mask = torch.triu(
            torch.ones(n_events, n_events, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        z = self.encoder(x, mask=causal_mask).squeeze(0)
        return self.head(z)

    def forward(self, batch) -> torch.Tensor:
        return self.forward_temporal(batch)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor, batch) -> torch.Tensor:
        return self.loss_fn(logits, labels, graph=batch)

    def training_step(self, batch, _idx):
        logits = self(batch)
        mask = self.scored_mask(batch)
        if not mask.any():
            return logits.sum() * 0.0
        labels = batch.y[mask].long()
        logits = logits[mask]
        loss = self._loss(logits, labels, batch)
        acc = (logits.argmax(1) == labels).float().mean()
        bs = int(labels.numel())
        self.log("train_loss", loss, batch_size=bs)
        self.log("train_acc", acc, batch_size=bs)
        return loss

    def validation_step(self, batch, _idx):
        logits = self(batch)
        mask = self.scored_mask(batch)
        if not mask.any():
            return None
        labels = batch.y[mask].long()
        logits = logits[mask]
        loss = self._loss(logits, labels, batch)
        probs = F.softmax(logits, dim=1)
        acc = (probs.argmax(1) == labels).float().mean()
        bs = int(labels.numel())
        self.log("val_loss", loss, batch_size=bs)
        self.log("val_acc", acc, batch_size=bs)
        if probs.shape[1] == 2:
            self._val_probs.append(probs[:, 1].detach().cpu())
            self._val_labels.append(labels.detach().cpu())
        return None

    def on_validation_epoch_end(self) -> None:
        if not self._val_probs:
            return
        labels = torch.cat(self._val_labels)
        if labels.unique().numel() >= 2:
            from torchmetrics.functional.classification import binary_auroc

            probs = torch.cat(self._val_probs)
            self.log("val_auroc", binary_auroc(probs, labels))
        self._val_probs.clear()
        self._val_labels.clear()

    def test_step(self, batch, _idx, dataloader_idx=0):
        logits = self(batch)
        mask = self.scored_mask(batch)
        if not mask.any():
            return None
        logits = logits[mask]
        labels = batch.y[mask].long()
        probs = F.softmax(logits, dim=1)
        attack_type = getattr(batch, "attack_type", None)
        self._record_test_batch(
            dataloader_idx,
            preds=probs.argmax(1),
            scores=probs,
            labels=labels,
            attack_type=attack_type[mask] if attack_type is not None else None,
        )
        return None

    def predict_step(self, batch, _idx):
        logits = self(batch)
        probs = F.softmax(logits, dim=1)
        out = {"preds": logits.argmax(1), "scores": probs[:, 1], "labels": batch.y}
        event_id = getattr(batch, "event_id", None)
        if event_id is not None:
            out["event_id"] = event_id
        return out
