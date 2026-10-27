"""Causal observation indices and training-only timing statistics."""

import numpy as np


def previous_rows(ids):
    """Previous physical row of each ID, independent of chunk boundaries."""
    order = np.argsort(ids, kind="stable")
    same = ids[order[1:]] == ids[order[:-1]]
    previous = np.full(len(ids), -1, dtype=np.int64)
    previous[order[1:][same]] = order[:-1][same]
    return previous


def _timing_statistics(gaps):
    """Scale, log median, log IQR, and training 99.9th-percentile clip."""
    positive = gaps[gaps > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    clip = max(float(np.quantile(gaps, 0.999)), scale)
    values = np.log1p(np.minimum(gaps, clip) / scale)
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    spread = float(q75 - q25)
    return scale, median, spread if spread >= 1e-6 else 1.0, clip


def fit_timing_statistics(recordings, *, min_support=32):
    """Fit only verified-normal train-query rows, never selection/target traffic.

    Deterministic evenly spaced rows cap sampling at 65,536 per recording.
    ID support counts refer to these sampled, observed predecessor pairs.
    Rows without a preceding bus/ID observation are excluded from fitting.
    """
    bus, gaps, ids = [], [], []
    for events in recordings:
        rec = events.recording
        if rec.role != "train_query" or not rec.verified_normal:
            continue
        times, event_ids = events.arrays["times"], events.arrays["ids"]
        rows = np.arange(1, events.count, max(1, (events.count + 65535) // 65536))
        if not len(rows):
            continue
        bus.append(times[rows] - times[rows - 1])
        previous = events.previous_rows
        if previous is None:
            previous = previous_rows(event_ids)
        rows = rows[previous[rows] >= 0]
        gaps.append(times[rows] - times[previous[rows]])
        ids.append(event_ids[rows])
    if not bus:
        raise ValueError("timing normalization requires verified-normal training pairs")
    bus = np.concatenate(bus)
    gaps = np.concatenate(gaps)
    ids = np.concatenate(ids)
    # A training cohort with no repeated IDs can still normalize bus timing.
    statistics = [
        _timing_statistics(bus),
        _timing_statistics(gaps if len(gaps) else bus),
    ]
    supported = []
    for key in np.unique(ids):
        selected = gaps[ids == key]
        if len(selected) >= min_support:
            supported.append(key)
            statistics.append(_timing_statistics(selected))
    return np.asarray(supported, dtype=np.int64), np.asarray(
        statistics, dtype=np.float64
    )
