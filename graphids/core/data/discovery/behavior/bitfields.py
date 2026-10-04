"""Calibrated, anonymous 12/16-bit CAN field candidates.

Reference captures set all numeric limits. A separate normal calibration capture
selects rules; normal test and attack files never enter rule selection.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from graphids.core.data.discovery.behavior.fields import file_hash, frames, sample_by_id


@dataclass(frozen=True)
class BitFieldConfig:
    samples_per_id: int = 1000
    min_frames: int = 80
    min_changes: int = 30
    min_distinct: int = 8
    max_q99_fraction: float = .04
    jump_multiplier: float = 4.0
    min_jump: int = 16
    calibration_min_transitions: int = 30
    calibration_max_violations: int = 0
    reference_max_violations: int = 0
    max_rules_per_id: int = 2
    bin_width: float = .1


def extract(payload: bytes, start: int, width: int, endian: str, signed: bool):
    total = len(payload) * 8
    if start + width > total:
        return None
    value = int.from_bytes(payload, endian)
    shift = total - start - width if endian == 'big' else start
    value = (value >> shift) & ((1 << width) - 1)
    if signed and value & (1 << (width - 1)):
        value -= 1 << width
    return value


def _quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _assess(payloads: list[bytes], spec: dict, cfg: BitFieldConfig):
    values = [extract(p, spec['start'], spec['width'], spec['endian'], spec['signed']) for p in payloads]
    if sum(v is not None for v in values) < cfg.min_frames:
        return None
    deltas = [abs(b - a) for a, b in zip(values, values[1:])
              if a is not None and b is not None and a != b]
    if len(deltas) < cfg.min_changes or len(set(v for v in values if v is not None)) < cfg.min_distinct:
        return None
    q99 = _quantile(deltas, .99)
    if q99 > cfg.max_q99_fraction * (1 << spec['width']):
        return None
    limit = max(cfg.min_jump, math.ceil(q99 * cfg.jump_multiplier))
    if limit >= (1 << spec['width']):
        return None
    return {'changes': len(deltas), 'q99_delta': q99, 'limit': limit,
            'fraction_changed': len(deltas) / max(1, len(values) - 1)}


def _specs(max_bytes: int):
    max_bits = min(64, max_bytes * 8)
    for width in (12, 16):
        for start in range(0, max_bits - width + 1, 4):
            for endian in ('big', 'little'):
                for signed in (False, True):
                    yield {'start': start, 'width': width, 'endian': endian, 'signed': signed}


def fit_references(paths: list[Path], cfg: BitFieldConfig) -> dict:
    if len(paths) != 2:
        raise ValueError('exactly two normal references required')
    samples = [sample_by_id(frames(path), cfg.samples_per_id) for path in paths]
    candidates = []
    for arb_id in sorted(set(samples[0]) & set(samples[1])):
        max_bytes = min(max(map(len, samples[0][arb_id])), max(map(len, samples[1][arb_id])))
        for spec in _specs(max_bytes):
            assessments = [_assess(s[arb_id], spec, cfg) for s in samples]
            if any(x is None for x in assessments):
                continue
            candidates.append({'arb_id': arb_id, **spec, 'limit': max(x['limit'] for x in assessments),
                               'reference': assessments})
    initial_count = len(candidates)
    reference_stats = [_rule_stats(frames(path), candidates) for path in paths]
    stable = []
    for index, candidate in enumerate(candidates):
        assessments = [stats[index] for stats in reference_stats]
        if all(item['violations'] <= cfg.reference_max_violations for item in assessments):
            stable.append({**candidate, 'reference_validation': assessments})
    return {'version': 2, 'config': asdict(cfg),
            'source_hashes': {str(path): file_hash(path) for path in paths},
            'initial_candidate_count': initial_count, 'candidates': stable}


def _rule_stats(rows, candidates: list[dict]):
    by_id = defaultdict(list)
    for index, rule in enumerate(candidates):
        by_id[rule['arb_id']].append((index, rule))
    previous = {}
    transitions = [0] * len(candidates)
    violations = [0] * len(candidates)
    for _stamp, _ordinal, arb_id, payload, _attack in rows:
        old = previous.get(arb_id)
        if old is not None and payload is not None:
            for index, rule in by_id[arb_id]:
                args = (rule['start'], rule['width'], rule['endian'], rule['signed'])
                a = extract(old, *args)
                b = extract(payload, *args)
                if a is None or b is None:
                    continue
                transitions[index] += 1
                violations[index] += abs(b - a) > rule['limit']
        previous[arb_id] = payload
    return [{'transitions': t, 'violations': v} for t, v in zip(transitions, violations)]


def calibrate(profile: dict, path: Path) -> dict:
    cfg = BitFieldConfig(**profile['config'])
    stats = _rule_stats(frames(path), profile['candidates'])
    eligible = []
    for rule, item in zip(profile['candidates'], stats):
        if item['transitions'] >= cfg.calibration_min_transitions and item['violations'] <= cfg.calibration_max_violations:
            eligible.append({**rule, 'calibration': item})
    # Prefer repeated activity in both references. Keep independent bit ranges per ID.
    eligible.sort(key=lambda r: (-min(x['changes'] for x in r['reference']),
                                 r['limit'] / (1 << r['width']), r['arb_id'], r['start'],
                                 r['width'], r['endian'], r['signed']))
    selected = []
    per_id = Counter()
    occupied = defaultdict(set)
    for rule in eligible:
        arb_id = rule['arb_id']
        bits = set(range(rule['start'], rule['start'] + rule['width']))
        if per_id[arb_id] >= cfg.max_rules_per_id or bits & occupied[arb_id]:
            continue
        selected.append(rule)
        per_id[arb_id] += 1
        occupied[arb_id] |= bits
    return {'version': 2, 'config': profile['config'], 'source_hashes': profile['source_hashes'],
            'calibration_hash': file_hash(path), 'candidate_count': len(profile['candidates']),
            'calibration_eligible_count': len(eligible), 'rules': selected}


def score(path: Path, profile: dict) -> dict:
    rows = frames(path)
    by_id = defaultdict(list)
    for rule in profile['rules']:
        by_id[rule['arb_id']].append(rule)
    previous = {}
    bins = defaultdict(lambda: {'field_jump': 0, 'attack_frames': 0, 'frames': 0})
    start = rows[0][0] if rows else 0.0
    end = rows[-1][0] if rows else 0.0
    covered = valid = invalid = 0
    for stamp, _ordinal, arb_id, payload, attack in rows:
        entry = bins[int((stamp - start) / profile['config']['bin_width'])]
        entry['frames'] += 1
        entry['attack_frames'] += int(attack)
        if payload is None:
            invalid += 1
            previous.pop(arb_id, None)
            continue
        if by_id[arb_id]:
            covered += 1
        old = previous.get(arb_id)
        if old is not None:
            for rule in by_id[arb_id]:
                args = (rule['start'], rule['width'], rule['endian'], rule['signed'])
                a, b = extract(old, *args), extract(payload, *args)
                if a is None or b is None:
                    continue
                valid += 1
                entry['field_jump'] += abs(b - a) > rule['limit']
        previous[arb_id] = payload
    table = [{'timestamp': min(end, start + (bin_id + 1) * profile['config']['bin_width']),
              **bins[bin_id]} for bin_id in sorted(bins)]
    alerts = sum(row['field_jump'] > 0 for row in table)
    attack_bins = sum(row['attack_frames'] > 0 for row in table)
    detected = sum(row['field_jump'] > 0 and row['attack_frames'] > 0 for row in table)
    hours = (end - start) / 3600
    return {'source': str(path), 'sha256': file_hash(path), 'duration_seconds': end - start,
            'rows': len(rows), 'invalid_payloads': invalid, 'covered_frames': covered,
            'covered_fraction': covered / len(rows) if rows else 0,
            'valid_rule_transitions': valid, 'alert_bins': alerts,
            'alert_bins_per_hour': alerts / hours if hours else None,
            'attack_bins': attack_bins, 'detected_attack_bins': detected,
            'attack_bin_recall': detected / attack_bins if attack_bins else None,
            'table': table}
