"""Stateless supervised classifier for temporal CAN events."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphids.core.losses import CrossEntropyLoss
from graphids.core.models.base import classification_test_metrics

from .base import TemporalModuleBase


class TemporalEventClassifier(TemporalModuleBase):
    """MLP baseline over ``TemporalData`` event messages plus ID embeddings."""

    _SCALES: dict[str, dict[str, int]] = {
        "small": {"hidden": 64, "layers": 2, "embedding_dim": 16},
        "large": {"hidden": 128, "layers": 3, "embedding_dim": 32},
    }

    def __init__(
        self,
        *,
        loss_fn: nn.Module | None = None,
        hidden: int | None = None,
        layers: int | None = None,
        embedding_dim: int | None = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scale: str = "small",
        model_type: str = "temporal_event_classifier",
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
        embedding_dim = embedding_dim if embedding_dim is not None else preset.get("embedding_dim", 16)
        super().__init__()
        self.test_metrics = classification_test_metrics(num_classes)
        self._val_probs: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._init_post(locals())

    def _build(self) -> None:
        hp = self.hparams
        self.src_embedding = nn.Embedding(max(1, int(hp.num_ids)), int(hp.embedding_dim))
        self.dst_embedding = nn.Embedding(max(1, int(hp.num_ids)), int(hp.embedding_dim))
        input_dim = int(hp.in_channels) + (2 * int(hp.embedding_dim))

        blocks: list[nn.Module] = []
        width = input_dim
        for _ in range(max(0, int(hp.layers) - 1)):
            blocks.extend(
                [
                    nn.Linear(width, int(hp.hidden)),
                    nn.ReLU(),
                    nn.Dropout(float(hp.dropout)),
                ]
            )
            width = int(hp.hidden)
        blocks.append(nn.Linear(width, int(hp.num_classes)))
        self.net = nn.Sequential(*blocks)
        self.test_metrics = classification_test_metrics(int(hp.num_classes))

    @staticmethod
    def _rebuild_excluded_kwargs(hp: dict) -> dict:
        from graphids.core.losses.build import build_loss

        return {"loss_fn": build_loss("temporal_event_classifier", hp.get("loss_config"))}

    def forward_temporal(self, batch, state=None) -> torch.Tensor:
        del state
        src = batch.src.clamp_min(0).clamp_max(self.src_embedding.num_embeddings - 1).long()
        dst = batch.dst.clamp_min(0).clamp_max(self.dst_embedding.num_embeddings - 1).long()
        features = torch.cat([batch.msg.float(), self.src_embedding(src), self.dst_embedding(dst)], dim=-1)
        return self.net(features)

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
