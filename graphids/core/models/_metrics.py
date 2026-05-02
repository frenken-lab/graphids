"""MetricCollection factories + custom Metric subclasses shared by every model family.

Factories (lazy torchmetrics import):

- ``classification_test_metrics(num_classes)`` — multiclass, used by GAT
  and fusion modules.
- ``binary_test_metrics(threshold=0.5)`` — binary, used by VGAE/DGI
  threshold-flavor and any binary classifier.

Custom Metric (eager torchmetrics import — required for class definition):

- ``BinaryYoudenJThreshold`` — pools scores/labels and returns the
  Youden-J threshold on ``compute()``.
"""

from __future__ import annotations

from typing import Any

import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import binary_roc
from torchmetrics.utilities.data import dim_zero_cat


class BinaryYoudenJThreshold(Metric):
    """Buffer pooled scores/labels; ``compute()`` returns the Youden-J threshold.

    Replaces the prior ``BinaryROC() + _find_threshold()`` pair. The
    ``.preds`` / ``.target`` list states are read directly by
    ``_log_thresholded_metrics`` (same attribute names as the old
    ``BinaryROC`` to preserve that access pattern).
    """

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


def classification_test_metrics(num_classes: int):
    """Unified classifier-flavor test metrics — aggregate, macro, weighted, per-class.

    Contract: ``update(probs, target)`` where ``probs`` is ``(N, K)`` float in the
    simplex and ``target`` is ``(N,)`` long. Decision metrics (accuracy/F1/
    precision/recall/specificity/MCC) argmax internally; curve metrics
    (AUROC/AP/ECE) consume the raw probabilities.

    Per-class decomposition uses ``ClasswiseWrapper`` — its dict return is
    merged flat into ``MetricCollection.compute()``, so ``log_dict(compute())``
    just works. Missing classes return 0 for their F1 (torchmetrics `#1494
    <https://github.com/Lightning-AI/torchmetrics/issues/1494>`_), which pulls
    the macro down; report ``weighted`` alongside ``macro``.
    """
    from torchmetrics import MetricCollection
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

    labels = (
        ["benign", "attack"] if num_classes == 2 else [f"class_{i}" for i in range(num_classes)]
    )
    k = {"num_classes": num_classes}

    def cw(metric, prefix):
        return ClasswiseWrapper(metric, labels=labels, prefix=prefix)

    return MetricCollection(
        {
            # Aggregate scalars — no per-class ambiguity.
            "accuracy": MulticlassAccuracy(**k, average="micro"),
            "mcc": MulticlassMatthewsCorrCoef(**k),
            "ece": MulticlassCalibrationError(**k),
            # Macro + weighted averages.
            "f1_macro": MulticlassF1Score(**k, average="macro"),
            "f1_weighted": MulticlassF1Score(**k, average="weighted"),
            "precision_macro": MulticlassPrecision(**k, average="macro"),
            "precision_weighted": MulticlassPrecision(**k, average="weighted"),
            "recall_macro": MulticlassRecall(**k, average="macro"),
            "recall_weighted": MulticlassRecall(**k, average="weighted"),
            "specificity_macro": MulticlassSpecificity(**k, average="macro"),
            "specificity_weighted": MulticlassSpecificity(**k, average="weighted"),
            "auc_macro": MulticlassAUROC(**k, average="macro"),
            "auc_weighted": MulticlassAUROC(**k, average="weighted"),
            "ap_macro": MulticlassAveragePrecision(**k, average="macro", thresholds=None),
            "ap_weighted": MulticlassAveragePrecision(**k, average="weighted", thresholds=None),
            # Per-class — ClasswiseWrapper expands into class-named keys.
            "f1_pc": cw(MulticlassF1Score(**k, average=None), "f1_per_class/"),
            "precision_pc": cw(MulticlassPrecision(**k, average=None), "precision_per_class/"),
            "recall_pc": cw(MulticlassRecall(**k, average=None), "recall_per_class/"),
            "specificity_pc": cw(
                MulticlassSpecificity(**k, average=None), "specificity_per_class/"
            ),
            "auc_pc": cw(MulticlassAUROC(**k, average=None), "auc_per_class/"),
            "ap_pc": cw(
                MulticlassAveragePrecision(**k, average=None, thresholds=None),
                "pr_auc_per_class/",
            ),
        }
    )


def binary_test_metrics(threshold: float = 0.5):
    """Standard binary classification MetricCollection shared by all modules.

    Contract: ``update(preds, target)`` expects ``preds`` to be a **float**
    tensor of probabilities in [0, 1] (or logits) and ``target`` to be
    long/int. Curve metrics (AUROC, AP, ECE) validate this and raise on int
    preds. The hard-pred metrics (accuracy/f1/precision/recall/specificity/
    MCC) apply ``threshold`` internally. Passing already-thresholded int
    preds is wrong: it crashes the curve metrics.

    ``threshold`` is forwarded to every metric that accepts it. For
    threshold-flavor models (VGAE/DGI), rebuild the collection after
    Youden-J discovery; for fusion models, pass the agent's
    ``decision_threshold`` at construction time.

    MCC is the chance-corrected summary for imbalanced binary data (CAN
    intrusion is ~1% positive). AP (area under PR curve) is the correct
    curve metric for imbalanced data. ECE measures probability calibration
    — meaningful only on classifier scores in [0, 1].
    """
    from torchmetrics import MetricCollection
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
    )

    return MetricCollection(
        {
            "accuracy": BinaryAccuracy(threshold=threshold),
            "f1": BinaryF1Score(threshold=threshold),
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "specificity": BinarySpecificity(threshold=threshold),
            "mcc": BinaryMatthewsCorrCoef(threshold=threshold),
            "auc": BinaryAUROC(),
            "ap": BinaryAveragePrecision(),
            "ece": BinaryCalibrationError(),
        }
    )
