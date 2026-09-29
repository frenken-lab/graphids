"""Lightning runtime hooks for the temporal hybrid model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class HybridRuntimeMixin:
    def on_train_epoch_start(self) -> None:
        self._train_state = None

    def on_validation_epoch_start(self) -> None:
        self._val_state = None
        self._val_cls_probs.clear()
        self._val_cls_labels.clear()
        self._val_anom_scores.clear()
        self._val_anom_labels.clear()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self._test_states = {}

    def training_step(self, batch, _idx):
        out, state = self._forward_with_state(batch, self._detach_state(self._train_state))
        self._train_state = self._detach_state(state)
        terms = self._loss_terms(out, batch)
        mask = self.scored_mask(batch)
        bs = max(1, int(mask.sum().item()))
        self.log("train_loss", terms["total"], batch_size=bs)
        for name, value in terms.items():
            if name in {"total", "zero"}:
                continue
            self.log(f"train_{name}_loss", value, batch_size=bs)
        if self.classification_enabled and "logits" in out and mask.any():
            labels = batch.y.reshape(-1)[mask].long()
            acc = (out["logits"][mask].argmax(1) == labels).float().mean()
            self.log("train_acc", acc, batch_size=int(labels.numel()))
        return terms["total"]

    def validation_step(self, batch, _idx):
        out, state = self._forward_with_state(batch, self._detach_state(self._val_state))
        self._val_state = self._detach_state(state)
        terms = self._loss_terms(out, batch)
        mask = self.scored_mask(batch)
        if not mask.any():
            return None
        labels = batch.y.reshape(-1)[mask].long()
        bs = int(labels.numel())
        self.log("val_loss", terms["total"], batch_size=bs)
        if self.classification_enabled and "logits" in out:
            probs = F.softmax(out["logits"][mask], dim=1)
            self.log("val_acc", (probs.argmax(1) == labels).float().mean(), batch_size=bs)
            if probs.shape[1] == 2:
                self._val_cls_probs.append(probs[:, 1].detach().cpu())
                self._val_cls_labels.append(labels.detach().cpu())
        if self.anomaly_enabled and "anomaly_score" in out:
            self._val_anom_scores.append(out["anomaly_score"][mask].detach().cpu())
            self._val_anom_labels.append(labels.detach().cpu())
        return None

    def on_validation_epoch_end(self) -> None:
        from torchmetrics.functional.classification import binary_auroc

        if self._val_cls_probs:
            labels = torch.cat(self._val_cls_labels)
            if labels.unique().numel() >= 2:
                self.log("val_cls_auroc", binary_auroc(torch.cat(self._val_cls_probs), labels))
        if self._val_anom_scores:
            labels = torch.cat(self._val_anom_labels)
            if labels.unique().numel() >= 2:
                self.log("val_anomaly_auroc", binary_auroc(torch.cat(self._val_anom_scores), labels))
        self._val_cls_probs.clear()
        self._val_cls_labels.clear()
        self._val_anom_scores.clear()
        self._val_anom_labels.clear()

    def test_step(self, batch, _idx, dataloader_idx=0):
        out, state = self._forward_with_state(
            batch,
            self._detach_state(self._test_states.get(dataloader_idx)),
        )
        self._test_states[dataloader_idx] = self._detach_state(state)
        mask = self.scored_mask(batch)
        if not mask.any():
            return None
        labels = batch.y.reshape(-1)[mask].long()
        attack_type = getattr(batch, "attack_type", None)
        attack_type = attack_type.reshape(-1)[mask] if attack_type is not None else None
        if self.classification_enabled and "logits" in out:
            probs = F.softmax(out["logits"][mask], dim=1)
            self._record_test_batch(
                dataloader_idx,
                preds=probs.argmax(1),
                scores=probs,
                labels=labels,
                attack_type=attack_type,
            )
        if self.anomaly_enabled and "anomaly_score" in out:
            self._record_binary_score_batch(
                dataloader_idx,
                scores=out["anomaly_score"][mask],
                labels=labels,
                attack_type=attack_type,
            )
        return None

    def predict_step(self, batch, _idx):
        out = self(batch)
        result: dict[str, torch.Tensor] = {"labels": batch.y.reshape(-1)}
        if "logits" in out:
            probs = F.softmax(out["logits"], dim=1)
            result["preds"] = probs.argmax(1)
            result["class_scores"] = probs[:, 1] if probs.shape[1] == 2 else probs
        if "anomaly_score" in out:
            result["scores"] = out["anomaly_score"]
        event_id = getattr(batch, "event_id", None)
        if event_id is not None:
            result["event_id"] = event_id.reshape(-1)
        return result
