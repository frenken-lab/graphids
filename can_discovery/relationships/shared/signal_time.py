"""Observation extraction and causal sample-and-hold in physical time."""
from __future__ import annotations

import math

import numpy as np

from ...decoding.field_values import extract
from ...decoding.signals import FieldCandidate
from ...preprocessing.capture import Capture


def observations(capture: Capture, field: FieldCandidate) -> tuple[np.ndarray, np.ndarray]:
    times = []
    values = []
    for frame in capture.frames:
        if frame.arb_id != field.arb_id or frame.payload is None:
            continue
        if field.bus is not None and (frame.bus or capture.known_bus) != field.bus:
            continue
        if len(frame.payload) != field.payload_length:
            continue
        if field.mux_start is not None and frame.payload[field.mux_start // 8] != field.mux_value:
            continue
        value = extract(frame.payload, field.start, field.width, field.endian, field.signed)
        if value is not None:
            times.append(frame.timestamp)
            values.append(value)
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def align(capture: Capture, fields: tuple[FieldCandidate, ...], *, step: float = .1,
          max_age: float = .5, max_bins: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Causal sample-and-hold with observation masks and age in physical time."""
    if step <= 0 or max_age < 0:
        raise ValueError("step must be positive and max_age nonnegative")
    if capture.end <= capture.start:
        raise ValueError("capture has no time span")
    step = max(step, (capture.end - capture.start) / max_bins)
    grid = (capture.start + step / 2 +
            np.arange(math.floor((capture.end - capture.start) / step)) * step)
    values = np.zeros((len(grid), len(fields)), dtype=float)
    valid = np.zeros_like(values, dtype=bool)
    ages = np.full_like(values, np.inf)
    for column, field in enumerate(fields):
        times, observed = observations(capture, field)
        if not len(times):
            continue
        indices = np.searchsorted(times, grid, side="right") - 1
        mask = indices >= 0
        ages[mask, column] = grid[mask] - times[indices[mask]]
        mask &= ages[:, column] <= max_age
        valid[:, column] = mask
        values[mask, column] = observed[indices[mask]]
    return values, valid, ages, step
