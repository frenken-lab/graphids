"""MetricCollection factories shared by temporal classifier models."""

from __future__ import annotations

from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
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
from torchmetrics.wrappers import ClasswiseWrapper


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
