"""Field histories, temporal fingerprints, motifs, and predictive graph edges."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from ..decoding.field_values import extract
from ..decoding.signals import FieldCandidate
from ..preprocessing.capture import Capture


@dataclass(frozen=True, slots=True)
class FieldEdge:
    source: str
    target: str
    relation: str
    lag_seconds: float
    effect: float
    q_value: float
    support: tuple[int, ...]
    step_seconds: float
    provenance: tuple[str, ...]


def field_key(field: FieldCandidate) -> str:
    mux = "" if field.mux_value is None else f"/m{field.mux_value}"
    return f"{field.arb_id}/{field.payload_length}/{field.start}:{field.width}:{field.endian}{mux}"


from .signal_time import observations, align


def _regular_history(capture: Capture, field: FieldCandidate,
                     max_samples: int = 4096) -> tuple[np.ndarray, float, int]:
    times, observed = observations(capture, field)
    if len(times) < 32:
        return np.empty(0), 0.0, len(observed)
    positive = np.diff(times)
    positive = positive[positive > 0]
    if not len(positive):
        return np.empty(0), 0.0, len(observed)
    step = max(float(np.median(positive)), (times[-1] - times[0]) / max_samples)
    grid = times[0] + step / 2 + np.arange(min(max_samples,
                    math.floor((times[-1] - times[0]) / step))) * step
    indices = np.searchsorted(times, grid, side="right") - 1
    valid = indices >= 0
    valid[valid] &= (grid[valid] - times[indices[valid]]) <= 1.5 * step
    edges = np.diff(np.r_[False, valid, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return np.empty(0), step, len(observed)
    best = int(np.argmax(ends - starts))
    values = observed[indices[starts[best]:ends[best]]]
    return values, step, len(observed)


def fingerprints(*args, **kwargs):
    from .experimental_analysis import fingerprints as run
    return run(*args, **kwargs)


def shape_motifs(*args, **kwargs):
    from .experimental_analysis import shape_motifs as run
    return run(*args, **kwargs)


def predictive_edges(*args, **kwargs):
    from .experimental_analysis import predictive_edges as run
    return run(*args, **kwargs)


def edges_to_dict(edges: tuple[FieldEdge, ...]) -> list[dict]:
    return [asdict(edge) for edge in edges]
