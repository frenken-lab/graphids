"""Score-based anomaly detector contract.

Models that produce a per-graph anomaly score (VGAE, DGI) inherit
:class:`ScoreBasedDetectorMixin` to get standard test-time plumbing:
ROC-curve threshold discovery (Youden-J), per-set + aggregate
threshold-flavor metrics, prediction recording, and checkpoint
persistence of the discovered threshold.

The leaf model implements ONE primitive: ``score(batch) -> Tensor``
returning one anomaly score per graph (higher = more anomalous).
Everything else — `test_step`, `on_test_epoch_end`, `predict_step`,
metric init — is inherited.
"""

from __future__ import annotations

import torch

from .base import GraphModuleBase, binary_test_metrics


class ScoreBasedDetectorMixin(GraphModuleBase):
    """Mix-in for graph models that produce per-graph anomaly scores.

    Inheriting this mixin in place of :class:`GraphModuleBase` swaps the
    classifier-flavor test path (per-set ``(N, K)`` probability metrics)
    for the threshold-flavor path (Youden-J discovery on pooled scores,
    binary metrics rebuilt at the discovered threshold).

    Subclasses MUST implement :meth:`score`. They MUST NOT override
    ``test_step`` / ``on_test_epoch_end`` / ``predict_step`` — those
    are the mixin's responsibility.
    """

    def __init__(self) -> None:
        super().__init__()
        self._init_threshold_metrics()
        self.test_metrics = binary_test_metrics()

    # -- contract -----------------------------------------------------------

    def score(self, batch) -> torch.Tensor:
        """Per-graph anomaly score, higher = more anomalous."""
        raise NotImplementedError

    # -- test lifecycle -----------------------------------------------------

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0) -> None:
        scores = self.score(batch)
        self.roc_metric.update(scores.detach(), batch.y.detach())
        self._record_test_batch(dataloader_idx, scores=scores, labels=batch.y)

    def on_test_epoch_end(self) -> None:
        # _log_thresholded_metrics (on GraphModuleBase) discovers the
        # threshold from self.roc_metric, rebuilds binary_test_metrics
        # at that threshold, logs per-set + aggregate + operating points,
        # and calls _finalize_test_predictions.
        self._log_thresholded_metrics()

    def predict_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        return {"scores": self.score(batch), "labels": batch.y}
