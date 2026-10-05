"""Reference-only broad payload behavior; signal recovery lives in signals."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .preprocessing.capture import Capture


@dataclass(frozen=True, slots=True)
class PayloadRule:
    arb_id: str
    reference_transitions: tuple[int, ...]
    hamming_limit: float
    change_rate: float
    run_limit: float
    changed_bit_fraction: tuple[float, ...]


def _quantile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(len(values) - 1, math.ceil(fraction * len(values)) - 1)]


def _groups(capture: Capture) -> dict[str, list[bytes]]:
    groups: dict[str, list[bytes]] = defaultdict(list)
    for frame in capture.frames:
        if frame.payload is not None:
            groups[frame.arb_id].append(frame.payload)
    return groups


def _changes(values: list[bytes]) -> tuple[list[int], list[int], int]:
    distances: list[int] = []
    flips = [0] * 64
    longest_run = current_run = 1
    for before, after in zip(values, values[1:]):
        if before == after:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
        if len(before) != len(after):
            continue
        delta = int.from_bytes(before, 'big') ^ int.from_bytes(after, 'big')
        distances.append(delta.bit_count())
        for bit in range(min(64, len(before) * 8)):
            flips[bit] += bool(delta & (1 << bit))
    return distances, flips, longest_run


def fit_payload(captures: list[Capture], *, min_transitions: int = 20) -> tuple[PayloadRule, ...]:
    groups = [_groups(capture) for capture in captures]
    rules = []
    for arb_id in sorted(set.intersection(*(set(group) for group in groups))):
        observations = [_changes(group[arb_id]) for group in groups]
        if any(len(item[0]) < min_transitions for item in observations):
            continue
        distances = [d for item in observations for d in item[0]]
        counts = tuple(len(item[0]) for item in observations)
        bit_counts = [sum(item[1][bit] for item in observations) for bit in range(64)]
        rules.append(PayloadRule(arb_id, counts, _quantile(distances, .995),
                                 sum(d > 0 for d in distances) / len(distances),
                                 max(item[2] for item in observations),
                                 tuple(n / sum(counts) for n in bit_counts)))
    return tuple(rules)
