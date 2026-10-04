"""Reference-only broad payload behavior and anonymous field hypotheses."""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from .bitfields import extract
from .capture import Capture


@dataclass(frozen=True, slots=True)
class PayloadRule:
    arb_id: str
    reference_transitions: tuple[int, ...]
    hamming_limit: float
    change_rate: float
    run_limit: float
    changed_bit_fraction: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FieldCandidate:
    arb_id: str
    start: int
    width: int
    endian: str
    signed: bool
    role: str
    limit: float
    confidence: float
    reference_support: tuple[int, ...]
    label_hypotheses: tuple[tuple[str, float], ...]
    evidence: tuple[str, ...]


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


def _field_assessment(values: list[bytes], spec: tuple[int, int, str, bool], min_changes: int):
    observed = [extract(payload, *spec) for payload in values]
    pairs = [(a, b) for a, b in zip(observed, observed[1:]) if a is not None and b is not None]
    changes = [b - a for a, b in pairs if a != b]
    if len(changes) < min_changes or len({v for v in observed if v is not None}) < 6:
        return None
    width = spec[1]
    if width <= 8:
        modulus = 1 << width
        counter_fraction = sum(delta % modulus == 1 for delta in changes) / len(changes)
        if counter_fraction >= .97:
            return ('counter', 0.0, len(changes), counter_fraction)
    absolute = [abs(delta) for delta in changes]
    q99 = _quantile(absolute, .99)
    if q99 >= .04 * (1 << width) or statistics.median(absolute) > .01 * (1 << width):
        return None
    return ('smooth', float(max(2, math.ceil(q99 * 4))), len(changes), 1 - q99 / (1 << width))


def _labels(role: str, spec: tuple[int, int, str, bool], stability: float) -> tuple[tuple[str, float], ...]:
    # These are behavior-based hypotheses, not calibrated probabilities or decoded signals.
    _, width, _, signed = spec
    if role == 'counter':
        return (('counter', stability),)
    if signed:
        return (('steering_angle', stability * .65), ('unknown', 1 - stability * .65))
    if width >= 12:
        return (('speed', stability * .55), ('rpm', stability * .50), ('unknown', 1 - stability * .55))
    return (('unknown', 1.0),)


def fit_fields(captures: list[Capture], *, max_ids: int = 64, samples_per_id: int = 512,
               min_changes: int = 20, max_fields_per_id: int = 4) -> tuple[FieldCandidate, ...]:
    """Search alternative boundaries, then validate each proposal on full references."""
    if len(captures) < 2:
        raise ValueError('field discovery needs independent normal references')
    groups = [_groups(capture) for capture in captures]
    ids = sorted(set.intersection(*(set(group) for group in groups)),
                 key=lambda key: (-sum(len(group[key]) for group in groups), key))[:max_ids]
    result = []
    for arb_id in ids:
        width_bits = min(64, *(max(map(len, group[arb_id])) * 8 for group in groups))
        proposals = []
        sampled_proposals = []
        for width in (4, 8, 12, 16, 24, 32):
            for start in range(0, width_bits - width + 1, 4):
                for endian in ('big', 'little'):
                    for signed in (False, True):
                        spec = (start, width, endian, signed)
                        sampled = [_field_assessment(group[arb_id][:samples_per_id], spec, min_changes)
                                   for group in groups]
                        if any(item is None for item in sampled):
                            continue
                        if len({item[0] for item in sampled}) != 1:
                            continue
                        sampled_proposals.append((min(item[3] for item in sampled), spec, sampled[0][0]))
        # Full-reference checks are intentionally capped for long recordings.
        sampled_proposals.sort(key=lambda item: (-item[0], item[1]))
        balanced = []
        per_shape = Counter()
        for proposal in sampled_proposals:
            shape = (proposal[1][1], proposal[2])
            if per_shape[shape] >= 8:
                continue
            balanced.append(proposal)
            per_shape[shape] += 1
        for _quality, spec, sampled_role in balanced:
            start, width, endian, signed = spec
            full = [_field_assessment(group[arb_id], spec, min_changes) for group in groups]
            if any(item is None or item[0] != sampled_role for item in full):
                continue
            role = full[0][0]
            limits = [item[1] for item in full]
            support = tuple(item[2] for item in full)
            stability = min(item[3] for item in full)
            if role == 'smooth':
                # A late reference jump invalidates a strict field hypothesis.
                observed = [[extract(payload, *spec) for payload in group[arb_id]] for group in groups]
                if any(any(abs(b - a) > limit for a, b in zip(values, values[1:])
                           if a is not None and b is not None)
                       for values, limit in zip(observed, limits, strict=True)):
                    continue
            confidence = min(1.0, max(0.0, stability))
            proposals.append(FieldCandidate(arb_id, start, width, endian, signed, role,
                                            max(limits), confidence, support,
                                            _labels(role, spec, confidence),
                                            ('independent_reference_support', 'full_reference_validation',
                                             'behavioral_name_only')))
        proposals.sort(key=lambda rule: (-rule.confidence, -min(rule.reference_support),
                                         rule.start, -rule.width, rule.endian, rule.signed))
        # Keep a few alternatives; do not declare one physical decoding canonical.
        selected = []
        per_range = Counter()
        for proposal in proposals:
            bit_range = (proposal.start, proposal.width)
            if per_range[bit_range] >= 2:
                continue
            selected.append(proposal)
            per_range[bit_range] += 1
            if len(selected) >= max_fields_per_id:
                break
        result.extend(selected)
    return tuple(result)
