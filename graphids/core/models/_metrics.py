"""MetricCollection factories + custom Metrics shared by every model family.

Both factories read from a single :data:`REGISTRY` of metric specs so
binary and multiclass collections stay in lockstep — adding a metric is
one row, not nine sites. Custom Metrics live alongside the factories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class MetricSpec:
    """``threshold``: binary supports ``threshold=`` kwarg.
    ``averaged``: multiclass fans out to macro/weighted + per-class.
    ``aggregate_avg``: multiclass ``average=`` for non-averaged metrics
    (``"micro"`` for accuracy; ``None`` to omit). ``extra``: kwargs
    forwarded to both ctors (e.g. ``thresholds=None`` for AP)."""

    name: str
    binary_cls: type
    multi_cls: type
    threshold: bool = True
    averaged: bool = True
    aggregate_avg: str | None = None
    extra: dict = field(default_factory=dict)


REGISTRY: tuple[MetricSpec, ...] = (
    MetricSpec("accuracy", BinaryAccuracy, MulticlassAccuracy, averaged=False, aggregate_avg="micro"),
    MetricSpec("f1", BinaryF1Score, MulticlassF1Score),
    MetricSpec("precision", BinaryPrecision, MulticlassPrecision),
    MetricSpec("recall", BinaryRecall, MulticlassRecall),
    MetricSpec("specificity", BinarySpecificity, MulticlassSpecificity),
    MetricSpec("mcc", BinaryMatthewsCorrCoef, MulticlassMatthewsCorrCoef, averaged=False),
    MetricSpec("auc", BinaryAUROC, MulticlassAUROC, threshold=False),
    MetricSpec("ap", BinaryAveragePrecision, MulticlassAveragePrecision, threshold=False, extra={"thresholds": None}),
    MetricSpec("ece", BinaryCalibrationError, MulticlassCalibrationError, threshold=False, averaged=False),
)


def binary_test_metrics(threshold: float = 0.5) -> MetricCollection:
    """Binary collection. ``preds`` must be float in [0, 1]; hard-pred
    metrics apply ``threshold`` internally. Rebuild after Youden-J for
    threshold-flavor models; pass ``decision_threshold`` for fusion."""
    items: dict = {}
    for s in REGISTRY:
        kw = {**s.extra, **({"threshold": threshold} if s.threshold else {})}
        items[s.name] = s.binary_cls(**kw)
    return MetricCollection(items)


def classification_test_metrics(num_classes: int) -> MetricCollection:
    """Multiclass collection: aggregate scalar + macro/weighted + per-class.
    ``probs`` is ``(N, K)`` float. ``ClasswiseWrapper`` flat-merges per-class
    keys. Missing classes return 0 F1 (torchmetrics #1494) — report
    ``weighted`` alongside ``macro``."""
    labels = ["benign", "attack"] if num_classes == 2 else [f"class_{i}" for i in range(num_classes)]
    items: dict = {}
    for s in REGISTRY:
        base = {"num_classes": num_classes, **s.extra}
        if not s.averaged:
            avg_kw = {"average": s.aggregate_avg} if s.aggregate_avg else {}
            items[s.name] = s.multi_cls(**base, **avg_kw)
            continue
        for avg in ("macro", "weighted"):
            items[f"{s.name}_{avg}"] = s.multi_cls(**base, average=avg)
        items[f"{s.name}_pc"] = ClasswiseWrapper(
            s.multi_cls(**base, average=None), labels=labels, prefix=f"{s.name}_per_class/"
        )
    return MetricCollection(items)
