"""Shared model infrastructure for temporal modules."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn


def strip_orig_mod_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """Drop ``_orig_mod.`` prefixes injected by ``torch.compile``."""
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


class _ModelBase(pl.LightningModule):
    """Shared Lightning utilities used by temporal modules."""

    def _store_init_kwargs(self, locals_dict: dict) -> None:
        import inspect

        sig = inspect.signature(type(self).__init__)
        names = tuple(n for n in sig.parameters if n != "self")
        saved: dict[str, Any] = {}
        for n in names:
            if n in locals_dict:
                v = locals_dict[n]
                setattr(self, n, v)
                if not isinstance(v, nn.Module):
                    saved[n] = v
        self.save_hyperparameters(saved)

    def prepare_from_datamodule(self, dm) -> None:
        tests = getattr(dm, "test_data", None) or getattr(dm, "test_datasets", None)
        self._test_set_names = list(tests.keys()) if tests else ["test"]
        self._attack_type_names = dict(getattr(dm, "attack_type_names", {0: "benign"}))

    def on_test_setup(self, datamodule, device) -> None:
        """Called before the test loop after the model is loaded."""

    def on_test_epoch_start(self) -> None:
        if hasattr(self, "test_metrics"):
            self.test_metrics.reset()
        names = getattr(self, "_test_set_names", None) or ["test"]
        if hasattr(self, "test_metrics"):
            self._per_set_metrics = {n: self.test_metrics.clone(prefix=f"test/{n}/") for n in names}
        self._test_buffers = {
            n: {"preds": [], "scores": [], "labels": [], "attack_type": []} for n in names
        }
        self._test_predictions: dict[str, dict[str, torch.Tensor]] = {}

    def _record_test_batch(
        self, dataloader_idx: int, *, scores, labels, preds=None, attack_type=None
    ) -> None:
        names = getattr(self, "_test_set_names", ["test"])
        name = names[dataloader_idx] if dataloader_idx < len(names) else names[-1]
        buf = self._test_buffers[name]
        buf["scores"].append(scores.detach().cpu())
        buf["labels"].append(labels.detach().cpu())
        if preds is not None:
            buf["preds"].append(preds.detach().cpu())
        if attack_type is not None:
            buf["attack_type"].append(attack_type.detach().cpu())

    def on_test_epoch_end(self) -> None:
        self._log_classifier_metrics()
        self._finalize_test_predictions()

    def _log_classifier_metrics(self) -> None:
        if not getattr(self, "_per_set_metrics", None):
            return
        self.test_metrics = self.test_metrics.cpu()
        all_probs, all_labels = [], []
        for name, coll in self._per_set_metrics.items():
            buf = self._test_buffers[name]
            if not buf["scores"]:
                continue
            probs = torch.cat(buf["scores"]).float()
            labels = torch.cat(buf["labels"]).long()
            coll = coll.cpu()
            self._per_set_metrics[name] = coll
            coll.update(probs, labels)
            self.log_dict(coll.compute())
            if probs.ndim == 2 and probs.shape[1] == 2:
                class1 = probs[:, 1]
                self._log_operating_points(class1, labels, prefix=f"test/{name}/")
                if buf["attack_type"]:
                    self._log_per_attack_auroc(name, class1, labels, torch.cat(buf["attack_type"]))
            all_probs.append(probs)
            all_labels.append(labels)
        if all_probs:
            pooled_p, pooled_l = torch.cat(all_probs), torch.cat(all_labels)
            self.test_metrics.update(pooled_p, pooled_l)
            self.log_dict(self.test_metrics.compute())
            if pooled_p.ndim == 2 and pooled_p.shape[1] == 2:
                self._log_operating_points(pooled_p[:, 1], pooled_l, prefix="test/")

    def _log_per_attack_auroc(
        self,
        name: str,
        class1_scores: torch.Tensor,
        labels: torch.Tensor,
        attack_type: torch.Tensor,
    ) -> None:
        if attack_type.numel() == 0:
            return
        from torchmetrics.functional.classification import binary_auroc

        scores = class1_scores.float()
        labels = labels.long()
        attack_type = attack_type.long()
        benign_mask = attack_type == 0
        names_map = getattr(self, "_attack_type_names", {0: "benign"})
        prefix = f"test/{name}/auroc_per_attack"
        per_attack: dict[str, float] = {}
        for code in attack_type.unique().tolist():
            if code == 0:
                continue
            subset = benign_mask | (attack_type == code)
            sub_scores = scores[subset]
            sub_labels = labels[subset]
            if sub_labels.unique().numel() < 2:
                continue
            attack_name = names_map.get(int(code), f"unknown_{int(code)}")
            per_attack[f"{prefix}/{attack_name}"] = float(binary_auroc(sub_scores, sub_labels))
        if not per_attack:
            return
        per_attack[f"test/{name}/auroc_per_attack_macro"] = sum(per_attack.values()) / len(per_attack)
        self.log_dict(per_attack)

    def _log_operating_points(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        *,
        prefix: str = "",
        min_recall: float = 0.95,
        min_precision: float = 0.99,
    ) -> None:
        if labels.unique().numel() < 2:
            return
        from torchmetrics.functional.classification import (
            binary_precision_at_fixed_recall,
            binary_recall_at_fixed_precision,
        )

        prec, thr_p = binary_precision_at_fixed_recall(scores, labels, min_recall=min_recall)
        rec, thr_r = binary_recall_at_fixed_precision(scores, labels, min_precision=min_precision)
        candidates = {
            f"{prefix}precision_at_{min_recall:g}recall": float(prec),
            f"{prefix}threshold_at_{min_recall:g}recall": float(thr_p),
            f"{prefix}recall_at_{min_precision:g}precision": float(rec),
            f"{prefix}threshold_at_{min_precision:g}precision": float(thr_r),
        }
        self.log_dict({k: v for k, v in candidates.items() if not math.isnan(v)})

    def _finalize_test_predictions(self) -> None:
        if not getattr(self, "_test_buffers", None):
            return
        self._test_predictions = {
            name: {
                k: torch.cat(v) if v else torch.empty(0)
                for k, v in buf.items()
                if v
            }
            for name, buf in self._test_buffers.items()
            if buf["scores"]
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        cls = type(self)
        checkpoint["class_path"] = f"{cls.__module__}.{cls.__name__}"
        if "state_dict" in checkpoint:
            checkpoint["state_dict"] = strip_orig_mod_prefix(checkpoint["state_dict"])


from ._metrics import classification_test_metrics  # noqa: E402, F401


def safe_load_checkpoint(model_type: str, ckpt_path, *, map_location="cpu"):
    """Load a checkpoint using the class path stored in the checkpoint."""
    del model_type
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    from graphids._fs import atomic_load

    ckpt = atomic_load(ckpt_path, map_location=map_location, weights_only=True)
    dotted = ckpt.get("class_path")
    if not dotted:
        raise KeyError(f"Checkpoint {ckpt_path} missing 'class_path'.")
    module_path, cls_name = dotted.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    hp = dict(ckpt.get("hyper_parameters", {}))
    rebuild = getattr(cls, "_rebuild_excluded_kwargs", None)
    extra_kwargs: dict = rebuild(hp) if rebuild is not None else {}
    module = cls(**{**hp, **extra_kwargs})
    state_dict = strip_orig_mod_prefix(ckpt["state_dict"])
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss_fn.")}
    module.load_state_dict(state_dict)
    module.to(map_location)
    if hasattr(module, "on_load_checkpoint"):
        module.on_load_checkpoint(ckpt)
    return module
