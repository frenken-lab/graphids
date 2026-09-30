"""Lightning runtime hooks for the temporal hybrid model."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from structlog import get_logger

log = get_logger(__name__)


class HybridRuntimeMixin:
    def _phase_timing_cfg(self) -> dict:
        return dict(getattr(self.hparams, "profiling", {}) or {})

    def _phase_timing_enabled(self) -> bool:
        return bool(self.training and self._phase_timing_cfg().get("phase_timing", False))

    def _scan_subphase_timing_enabled(self) -> bool:
        scan_cfg = dict(getattr(self.hparams, "scan", {}) or {})
        return bool(self._phase_timing_enabled() and scan_cfg.get("profile_subphases", False))

    def _phase_sync_cuda(self) -> None:
        cfg = self._phase_timing_cfg()
        if bool(cfg.get("sync_cuda", True)) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _phase_start(self) -> float | None:
        if not self._phase_timing_enabled():
            return None
        self._phase_sync_cuda()
        return time.perf_counter()

    def _phase_end(self, name: str, started_at: float | None) -> None:
        if started_at is None:
            return
        self._phase_sync_cuda()
        elapsed = max(0.0, time.perf_counter() - started_at)
        timings = getattr(self, "_train_phase_timings_s", None)
        if not isinstance(timings, dict):
            timings = {}
            self._train_phase_timings_s = timings
        timings[name] = timings.get(name, 0.0) + elapsed

    def _scan_subphase_start(self) -> float | None:
        if not self._scan_subphase_timing_enabled():
            return None
        return self._phase_start()

    def _scan_subphase_end(self, name: str, started_at: float | None) -> None:
        self._phase_end(f"scan_{name}", started_at)

    def _should_log_phase_timings(self, batch_idx: int) -> bool:
        cfg = self._phase_timing_cfg()
        warmup = max(0, int(cfg.get("warmup_steps", 5)))
        every = max(1, int(cfg.get("log_every_n_steps", 10)))
        profiled = int(batch_idx) - warmup + 1
        return profiled > 0 and profiled % every == 0

    def _log_phase_timings(self, batch_idx: int) -> None:
        timings = dict(getattr(self, "_train_phase_timings_s", {}) or {})
        if not timings or not self._should_log_phase_timings(batch_idx):
            return
        cfg = self._phase_timing_cfg()
        prefix = str(cfg.get("prefix", "profile"))
        metrics = {f"{prefix}/train_phase_{name}_ms": value * 1000.0 for name, value in timings.items()}
        if "heads" in timings and "loss" in timings:
            metrics[f"{prefix}/train_phase_heads_loss_ms"] = (timings["heads"] + timings["loss"]) * 1000.0
        for name, value in metrics.items():
            self.log(name, value, on_step=True, on_epoch=False, logger=True, prog_bar=False)
        log.info(
            "temporal_phase_profile",
            prefix=prefix,
            batch_idx=batch_idx,
            **{name.rsplit("/", 1)[-1]: round(value, 3) for name, value in metrics.items()},
        )

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
        if self._phase_timing_enabled():
            self._train_phase_timings_s = {}
        forward_start = self._phase_start()
        out, state = self._forward_with_state(batch, self._detach_state(self._train_state))
        self._phase_end("forward", forward_start)
        self._train_state = self._detach_state(state)
        loss_start = self._phase_start()
        terms = self._loss_terms(out, batch)
        mask = self.scored_mask(batch)
        bs = max(1, int(mask.sum().item()))
        self._phase_end("loss", loss_start)
        self.log("train_loss", terms["total"], batch_size=bs)
        for name, value in terms.items():
            if name in {"total", "zero"}:
                continue
            self.log(f"train_{name}_loss", value, batch_size=bs)
        if self.classification_enabled and "logits" in out and mask.any():
            labels = batch.y.reshape(-1)[mask].long()
            acc = (out["logits"][mask].argmax(1) == labels).float().mean()
            self.log("train_acc", acc, batch_size=int(labels.numel()))
        self._log_phase_timings(int(_idx))
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
