"""Unary physical-role candidate scoring and capture sampling."""
from __future__ import annotations

import math

import numpy as np

from ...decoding.field_values import extract
from ...decoding.signals import FieldCandidate
from ...preprocessing.capture import Capture
from .role_types import RoleHypothesis


def _index_capture(capture: Capture) -> dict[tuple[str, int], list]:
    grouped: dict[tuple[str, int], list] = {}
    for frame in capture.frames:
        if frame.payload is not None:
            grouped.setdefault((frame.arb_id, len(frame.payload)), []).append(frame)
    return grouped


def _sample_values(capture: Capture, field: FieldCandidate, cap: int = 2048,
                   index: dict[tuple[str, int], list] | None = None) -> tuple[np.ndarray, np.ndarray]:
    frames = (index.get((field.arb_id, field.payload_length), []) if index is not None else
              [frame for frame in capture.frames if frame.arb_id == field.arb_id
               and frame.payload is not None and len(frame.payload) == field.payload_length])
    frames = [frame for frame in frames if (frame.bus or capture.known_bus) == field.bus]
    if field.mux_start is not None:
        frames = [frame for frame in frames if frame.payload[field.mux_start // 8] == field.mux_value]
    if len(frames) > cap:
        indices = np.linspace(0, len(frames) - 1, cap, dtype=int)
        frames = [frames[int(index)] for index in indices]
    times = np.asarray([frame.timestamp for frame in frames], dtype=float)
    values = np.asarray([extract(frame.payload, field.start, field.width,
                                 field.endian, field.signed) for frame in frames], dtype=float)
    return times, values


def _series_metrics(series: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple] | None:
    if min((len(values) for _, values in series), default=0) < 30:
        return None
    metrics = []
    for times, values in series:
        distinct = len(np.unique(values))
        changes = np.diff(values)
        duration = max(float(times[-1] - times[0]), 1e-6)
        spread = max(float(np.ptp(values)), 1.0)
        change_rate = np.count_nonzero(changes) / max(1, len(changes))
        jump = float(np.quantile(np.abs(changes), .95) / spread) if len(changes) else 1.0
        dwell = float(np.mean(changes == 0)) if len(changes) else 0.0
        zero_fraction = float(np.mean(values == 0))
        sign_cross = bool(np.min(values) < 0 < np.max(values))
        metrics.append((distinct, change_rate, jump, dwell, zero_fraction,
                        sign_cross, duration, float(np.max(values))))
    return metrics


def _role_score(role: str, series: list[tuple[np.ndarray, np.ndarray]],
                field: FieldCandidate, metrics: list[tuple] | None = None) -> tuple[float, tuple[str, ...]]:
    """Scale-free unary evidence. These scores are ranks, not probabilities."""
    metrics = _series_metrics(series) if metrics is None else metrics
    if metrics is None:
        return -math.inf, ()
    if role in ("speed", "rpm", "acceleration", "steering", "yaw", "lateral_acceleration"):
        # A normal trip can be parked or nearly straight; require excitation
        # on at least one reference and let factors mark the other unavailable.
        if max(item[0] for item in metrics) < 8:
            return -math.inf, ()
        smooth = min(max(0.0, 1.0 - item[2] * 3) for item in metrics)
        varying = min(min(1.0, item[0] / 32) for item in metrics)
        score = .40 * smooth + .25 * varying + .15 * field.confidence
        if role == "speed":
            score += .10 * min(1 - item[4] for item in metrics)
            evidence = ("continuous_motion", "bounded_updates")
        elif role == "rpm":
            score += .10 * min(item[1] for item in metrics)
            evidence = ("continuous_engine_like", "frequent_updates")
        elif role == "acceleration":
            score += .10 * min(item[5] for item in metrics)
            evidence = ("signed_dynamic", "bounded_updates")
        else:
            score += .10 * min(item[5] for item in metrics)
            evidence = ("signed_centered_dynamic", "bounded_updates")
    elif role == "gear":
        if max(item[0] for item in metrics) > 32 or max(item[0] for item in metrics) < 2:
            return -math.inf, ()
        # Gear may remain constant throughout a capture. One varying reference
        # is helpful, but not required for a provisional state hypothesis.
        score = (.35 * min(item[3] for item in metrics) +
                 .25 * min(min(1.0, 8 / max(1, item[0])) for item in metrics) +
                 .15 * field.confidence)
        if any(item[0] > 1 for item in metrics):
            score += .10
        evidence = ("finite_states", "long_dwell")
    else:
        if max(item[0] for item in metrics) < 2:
            return -math.inf, ()
        score = .30 * min(item[3] for item in metrics) + .20 * min(
            min(1.0, item[0] / 8) for item in metrics) + .15 * field.confidence
        evidence = (("sparse_braking", "persistent_activation") if role == "brake" else
                    ("throttle_modulation", "persistent_activation"))
    # These are explicitly specified role properties, independent of any
    # target DBC. Scale is unknown, so only sign, state count, update rate,
    # and a broad bit-width range can be used at this stage.
    widths = {
        "speed": (8, 16), "rpm": (10, 16), "gear": (5, 12),
        "acceleration": (8, 16), "steering": (10, 16),
        "yaw": (8, 16), "lateral_acceleration": (8, 16),
        "brake": (4, 16), "throttle": (6, 16),
    }
    low, high = widths[role]
    score += .14 if low <= field.width <= high else (-.18 if field.width > high + 4 else -.08)
    if role == "gear":
        if field.width < 3:
            return -math.inf, ()
        if max(item[0] for item in metrics) >= 3:
            score += .06
    elif role in ("speed", "rpm", "brake", "throttle"):
        score += .06 if not field.signed else -.10
        if any(item[5] for item in metrics):
            score -= .25
        if not field.signed and any(item[7] >= .75 * (1 << field.width) for item in metrics):
            score -= .20
    if role == "speed":
        score += .10 if any(item[4] > .02 for item in metrics) else -.03
    elif role == "rpm":
        score += .08 * min(item[1] for item in metrics)
    elif role in ("steering", "acceleration", "yaw", "lateral_acceleration"):
        score += .16 if field.signed and any(item[5] for item in metrics) else -.12
    elif role == "brake":
        score += .18 * min(item[4] for item in metrics)
    elif role == "throttle":
        score += .10 * min(item[1] for item in metrics)
    return float(score), evidence


def _field_id(field: FieldCandidate) -> tuple:
    return (field.arb_id, field.payload_length, field.start, field.width,
            field.endian, field.signed, field.mux_start, field.mux_value)


def _equivalence_groups(hypotheses: list[RoleHypothesis], captures: list[Capture],
                        indices: list[dict[tuple[str, int], list]] | None = None) -> list[int]:
    """Group affine-equivalent histories within one message family."""
    signatures = {}
    output = []
    for hypothesis in hypotheses:
        field = hypothesis.candidate
        traces = []
        for number, capture in enumerate(captures):
            _, values = _sample_values(capture, field, cap=512,
                                        index=indices[number] if indices is not None else None)
            if not len(values):
                continue
            centered = values - values[0]
            scale = np.max(np.abs(centered))
            traces.append(tuple(np.round(centered / scale, 8)) if scale else ("constant",))
        signature = (field.bus, field.arb_id, field.payload_length, field.mux_start,
                     field.mux_value, tuple(traces))
        output.append(signatures.setdefault(signature, len(signatures)))
    return output
