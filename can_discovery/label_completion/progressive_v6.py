"""Value classes and source-gated evidence for full-pool label completion."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from ..decoding.signals import FieldCandidate
from ..relationships.shared.signal_time import observations

SCHEMA = 'dbc-label-completion-v6-consolidated-v1'
CONTRAST_WEIGHT = .15
RELATION_WEIGHT = .15
RELATION_MIN_NET = .04
RELATION_SOURCE_VEHICLES = 2


def field_from_row(row: dict) -> FieldCandidate:
    return FieldCandidate(row['arb_id'], row['start'], row['width'],
        row['endian'], row['signed'], 'unknown', 0., .5, (128, 128),
        (('unknown', 1.),), ('full_pool_v5',), row['payload_length'],
        bus=row['bus'])


def decoded_values(row: dict, samples: list) -> list[tuple[np.ndarray, np.ndarray]]:
    shift = (row['payload_length'] * 8 - row['start'] - row['width']
             if row['endian'] == 'big' else row['start'])
    mask = (1 << row['width']) - 1
    result = []
    for times, little, big in samples:
        raw = ((big if row['endian'] == 'big' else little) >> shift) & mask
        values = raw.astype(np.float64)
        if row['signed']:
            sign = 1 << (row['width'] - 1)
            values = np.where(values >= sign, values - (1 << row['width']), values)
        result.append((times, values))
    return result


def trajectory_signature(series: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """Affine-equivalent sampled histories share a class, with no layout discarded."""
    digest = hashlib.sha256()
    for _, values in series:
        if not len(values):
            digest.update(b'empty')
            continue
        centered = values - values[0]
        scale = float(np.max(np.abs(centered)))
        if scale < 1e-10:
            digest.update(b'constant')
            continue
        normalized = np.round(centered / scale, 4).astype(np.float32)
        normalized[normalized == 0] = 0.
        forward = normalized.tobytes()
        reversed_values = -normalized
        reversed_values[reversed_values == 0] = 0.
        reverse = reversed_values.tobytes()
        digest.update(min(forward, reverse))
        digest.update(b'|')
    return digest.hexdigest()[:20]


def candidate_features(row: dict, series: list[tuple[np.ndarray, np.ndarray]],
                       unary_score: float) -> np.ndarray:
    """Scale-free features shared by source training and target ranking."""
    route_features = []
    for _, values in series:
        if len(values) < 2:
            return np.full(7, np.nan)
        changes = np.diff(values)
        span = max(float(np.ptp(values)), 1.)
        route_features.append((min(len(np.unique(values)), 64) / 64,
            float(np.count_nonzero(changes) / len(changes)),
            float(np.quantile(np.abs(changes), .95) / span),
            float(np.mean(values == 0))))
    distinct, change, jump, zeros = np.mean(route_features, axis=0)
    return np.asarray([unary_score, distinct, change, jump, zeros,
                       row['width'] / 64, float(row['signed'])], dtype=float)


@dataclass(frozen=True)
class ContrastModel:
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def adjustment(self, features: np.ndarray) -> float:
        if not np.isfinite(features).all():
            return 0.
        value = float(np.dot((features - self.center) / self.scale, self.weights))
        return CONTRAST_WEIGHT * float(np.clip(value, -1., 1.))


def fit_contrast(positives: list[np.ndarray], negatives: list[np.ndarray]
                 ) -> ContrastModel | None:
    if len(positives) < 2 or len(negatives) < 4:
        return None
    x = np.stack([*positives, *negatives])
    if not np.isfinite(x).all():
        return None
    center = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), .05)
    standardized = (x - center) / scale
    labels = np.r_[np.ones(len(positives)), -np.ones(len(negatives))]
    sample_weight = np.r_[np.full(len(positives), 1 / len(positives)),
                          np.full(len(negatives), 1 / len(negatives))]
    weighted_x = standardized * np.sqrt(sample_weight[:, None])
    weighted_y = labels * np.sqrt(sample_weight)
    weights = np.linalg.solve(weighted_x.T @ weighted_x + .5 * np.eye(x.shape[1]),
                              weighted_x.T @ weighted_y)
    return ContrastModel(center, scale, weights)


def relation_net(series: list[tuple[np.ndarray, np.ndarray]],
                 anchor_observations: list[tuple[np.ndarray, np.ndarray]]) -> float | None:
    """Synchronized change association beyond a route-local shifted control."""
    nets = []
    for (times, values), (anchor_times, anchor_values) in zip(
            series, anchor_observations, strict=True):
        if len(times) < 32 or len(anchor_times) < 32:
            return None
        indices = np.searchsorted(anchor_times, times, side='right') - 1
        valid = indices >= 0
        ages = np.full(len(times), np.inf)
        ages[valid] = times[valid] - anchor_times[indices[valid]]
        valid &= ages <= .3
        if valid.sum() < 32:
            return None
        candidate = values[valid]
        anchored = anchor_values[indices[valid]]
        x = np.diff(candidate)
        y = np.diff(anchored)
        if min(np.std(x), np.std(y)) < 1e-8:
            return None
        actual = float(np.corrcoef(x, y)[0, 1])
        shifted = float(np.corrcoef(x, np.roll(y, max(10, len(y)//3)))[0, 1])
        if not math.isfinite(actual) or not math.isfinite(shifted):
            return None
        nets.append(max(0., actual - shifted))
    return min(nets) if nets else None


def anchor_series(captures, field: FieldCandidate):
    return [observations(capture, field) for capture in captures]
