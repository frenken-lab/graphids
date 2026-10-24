"""Exact frame metrics shared by event evaluation."""

import numpy as np


def raw_average_precision(scores, labels):
    """Exact noninterpolated AP, grouping equal raw scores at their rank endpoint.

    Unknown labels (-1) are excluded. Single-class groups return None.
    No probability conversion is applied, even outside [0, 1].
    """
    return frame_metrics(scores, labels).get("frame_ap")


def frame_metrics(scores, labels, *, threshold=None):
    """Exact tied-score AP/AUC; optional diagnostic confusion matrix."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must be aligned vectors")
    if not np.isfinite(scores).all():
        raise ValueError("AP requires finite raw scores")
    if not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError("labels must be -1, 0 or 1")
    known = labels >= 0
    scores, positive = scores[known], labels[known] == 1
    p, n = int(positive.sum()), int((~positive).sum())
    metrics = dict(frames=p + n, positives=p, negatives=n)
    if p and n:
        order = np.argsort(-scores, kind="stable")
        ranked = scores[order]
        ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), len(ranked) - 1]
        tp = np.r_[0, np.cumsum(positive[order])[ends]]
        fp = np.r_[0, ends + 1 - tp[1:]]
        auc = np.sum(np.diff(fp / n) * (tp[1:] + tp[:-1]) / (2 * p))
        ap = np.sum(np.diff(tp) * tp[1:] / (ends + 1)) / p
        metrics.update(frame_ap=float(ap), frame_auroc=float(auc))
    if threshold is not None:
        predicted = scores >= threshold
        tp, fp = int((predicted & positive).sum()), int((predicted & ~positive).sum())
        fn, tn = p - tp, n - fp
        metrics.update(tp=tp, fp=fp, fn=fn, tn=tn)
        for key, numerator, denominator in (
            ("precision", tp, tp + fp),
            ("recall", tp, p),
            ("f1", 2 * tp, 2 * tp + fp + fn),
            ("fpr", fp, n),
            ("accuracy", tp + tn, p + n),
        ):
            metrics[key] = numerator / denominator if denominator else None
    return metrics
