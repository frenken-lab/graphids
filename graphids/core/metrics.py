"""Exact frame metrics shared by event evaluation."""

import numpy as np


def raw_average_precision(scores, labels):
    """Exact noninterpolated AP, grouping equal raw scores at their rank endpoint.

    Unknown labels (-1) are excluded. Single-class groups return None.
    No probability conversion is applied, even outside [0, 1].
    """
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must be aligned vectors")
    if not np.isfinite(scores).all():
        raise ValueError("AP requires finite raw scores")
    if not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError("labels must be -1, 0 or 1")
    known = labels >= 0
    scores, labels = scores[known], labels[known]
    if set(labels) != {0, 1}:
        return None
    order = np.argsort(-scores, kind="stable")
    scores, positives = scores[order], labels[order] == 1
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    tp = np.cumsum(positives)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / positives.sum())
