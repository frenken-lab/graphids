"""MetricCollection factories + custom metrics shared by every model family."""

from __future__ import annotations

from typing import Any

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)
from torchmetrics.functional.classification import binary_roc
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.wrappers import ClasswiseWrapper


class BinaryYoudenJThreshold(Metric):
    """Pools (preds, target); ``compute()`` returns the Youden-J threshold."""

    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        p = dim_zero_cat(self.preds)
        t = dim_zero_cat(self.target).long()
        fpr, tpr, thr = binary_roc(p, t)
        if thr.numel() < 2:
            return torch.tensor(float("nan"), device=thr.device)
        return thr[torch.argmax(tpr - fpr)]


def binary_test_metrics(threshold: float = 0.5) -> MetricCollection:
    """Binary collection. ``preds`` must be float in [0, 1]."""
    return MetricCollection(
        {
            "accuracy": BinaryAccuracy(threshold=threshold),
            "f1": BinaryF1Score(threshold=threshold),
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "specificity": BinarySpecificity(threshold=threshold),
            "mcc": BinaryMatthewsCorrCoef(threshold=threshold),
            "auroc": BinaryAUROC(),
            "ap": BinaryAveragePrecision(thresholds=None),
            "ece": BinaryCalibrationError(),
        }
    )


def classification_test_metrics(num_classes: int) -> MetricCollection:
    """Multiclass collection: aggregate scalar + macro/weighted + per-class."""
    labels = ["benign", "attack"] if num_classes == 2 else [f"class_{i}" for i in range(num_classes)]
    items: dict[str, Metric] = {
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "f1_weighted": MulticlassF1Score(num_classes=num_classes, average="weighted"),
        "f1_pc": ClasswiseWrapper(
            MulticlassF1Score(num_classes=num_classes, average=None),
            labels=labels,
            prefix="f1_per_class/",
        ),
        "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "precision_weighted": MulticlassPrecision(num_classes=num_classes, average="weighted"),
        "precision_pc": ClasswiseWrapper(
            MulticlassPrecision(num_classes=num_classes, average=None),
            labels=labels,
            prefix="precision_per_class/",
        ),
        "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
        "recall_weighted": MulticlassRecall(num_classes=num_classes, average="weighted"),
        "recall_pc": ClasswiseWrapper(
            MulticlassRecall(num_classes=num_classes, average=None),
            labels=labels,
            prefix="recall_per_class/",
        ),
        "specificity_macro": MulticlassSpecificity(num_classes=num_classes, average="macro"),
        "specificity_weighted": MulticlassSpecificity(num_classes=num_classes, average="weighted"),
        "specificity_pc": ClasswiseWrapper(
            MulticlassSpecificity(num_classes=num_classes, average=None),
            labels=labels,
            prefix="specificity_per_class/",
        ),
        "mcc": MulticlassMatthewsCorrCoef(num_classes=num_classes),
        "auroc_macro": MulticlassAUROC(num_classes=num_classes, average="macro"),
        "auroc_weighted": MulticlassAUROC(num_classes=num_classes, average="weighted"),
        "auroc_pc": ClasswiseWrapper(
            MulticlassAUROC(num_classes=num_classes, average=None),
            labels=labels,
            prefix="auroc_per_class/",
        ),
        "ap_macro": MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
        "ap_weighted": MulticlassAveragePrecision(num_classes=num_classes, average="weighted"),
        "ap_pc": ClasswiseWrapper(
            MulticlassAveragePrecision(num_classes=num_classes, average=None),
            labels=labels,
            prefix="ap_per_class/",
        ),
        "ece": MulticlassCalibrationError(num_classes=num_classes),
    }
    return MetricCollection(items)
