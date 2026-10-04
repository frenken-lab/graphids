"""Reference-only, byte-aligned CAN payload behavior pilot.

These are anonymous field roles, not decoded physical quantities.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FieldConfig:
    samples_per_id: int = 2000
    min_frames_per_reference: int = 50
    min_counter_transitions: int = 30
    min_numeric_transitions: int = 30
    counter_min_accuracy: float = .97
    numeric_jump_multiplier: float = 4.0
    numeric_min_jump: int = 8
    bin_width: float = .1


from .capture import file_hash, load_capture


def frames(path: Path):
    """Compatibility rows from the shared capture parser."""
    return [(f.timestamp, f.ordinal, f.arb_id, f.payload, f.attack)
            for f in load_capture(path).frames]


def sample_by_id(rows, cap: int):
    picked = defaultdict(list)
    for _stamp, _ordinal, arb_id, payload, _attack in rows:
        if payload is None:
            continue
        group = picked[arb_id]
        if len(group) < cap:
            group.append(payload)
    return picked


def _field_values(payloads, index: int, kind: str) -> list[int]:
    if kind == 'byte':
        return [p[index] for p in payloads if len(p) > index]
    shift = 4 if kind == 'high_nibble' else 0
    return [(p[index] >> shift) & 15 for p in payloads if len(p) > index]


def _candidate(payloads: list[bytes], index: int, kind: str, cfg: FieldConfig):
    values = _field_values(payloads, index, kind)
    if len(values) < cfg.min_frames_per_reference:
        return None
    changes = [(a, b) for a, b in zip(values, values[1:]) if a != b]
    if kind != 'byte':
        if len(changes) < cfg.min_counter_transitions:
            return None
        accuracy = sum((b - a) % 16 == 1 for a, b in changes) / len(changes)
        return {'accuracy': accuracy, 'transitions': len(changes)} if accuracy >= cfg.counter_min_accuracy else None
    if len(changes) < cfg.min_numeric_transitions or len(set(values)) < 8:
        return None
    deltas = [abs(b - a) for a, b in changes]
    # A field dominated by wraparound or random bytes is not a smooth numeric candidate.
    if statistics.median(deltas) > 16 or sum(d > 64 for d in deltas) / len(deltas) > .05:
        return None
    ordered = sorted(deltas)
    q99 = ordered[min(len(ordered) - 1, math.ceil(.99 * len(ordered)) - 1)]
    limit = max(cfg.numeric_min_jump, math.ceil(q99 * cfg.numeric_jump_multiplier))
    if limit >= 255:
        return None
    return {'limit': limit, 'transitions': len(changes),
            'median_delta': statistics.median(deltas), 'q99_delta': q99}


def fit(paths: list[Path], cfg: FieldConfig) -> dict:
    if len(paths) != 2:
        raise ValueError('exactly two frozen normal references required')
    sampled = [sample_by_id(frames(path), cfg.samples_per_id) for path in paths]
    rules = []
    inventory = Counter()
    for arb_id in sorted(set(sampled[0]) & set(sampled[1])):
        max_bytes = min(max(map(len, sampled[0][arb_id])), max(map(len, sampled[1][arb_id])))
        for index in range(max_bytes):
            for kind in ('high_nibble', 'low_nibble', 'byte'):
                candidates = [_candidate(file[arb_id], index, kind, cfg) for file in sampled]
                if any(c is None for c in candidates):
                    continue
                role = 'counter_candidate' if kind != 'byte' else 'smooth_byte'
                inventory[role] += 1
                rule = {'arb_id': arb_id, 'byte_index': index, 'kind': kind,
                        'role': role, 'reference': candidates}
                if kind == 'byte':
                    rule['limit'] = max(c['limit'] for c in candidates)
                rules.append(rule)
    return {'version': 1, 'config': asdict(cfg),
            'source_hashes': {str(p): file_hash(p) for p in paths},
            'sampled_ids': [len(s) for s in sampled], 'inventory': dict(inventory),
            'rules': rules}


def score(path: Path, profile: dict) -> dict:
    rows = frames(path)
    cfg = profile['config']
    bin_width = cfg['bin_width']
    rules = defaultdict(list)
    for rule in profile['rules']:
        rules[rule['arb_id']].append(rule)
    previous = {}
    bins = defaultdict(lambda: {'counter': 0, 'numeric_jump': 0, 'attack_frames': 0,
                                'frames': 0})
    covered = invalid = 0
    start = rows[0][0] if rows else 0.0
    end = rows[-1][0] if rows else 0.0
    for stamp, _ordinal, arb_id, payload, attack in rows:
        slot = int((stamp - start) / bin_width)
        entry = bins[slot]
        entry['frames'] += 1
        entry['attack_frames'] += int(attack)
        if payload is None:
            invalid += 1
            previous.pop(arb_id, None)
            continue
        applicable = rules[arb_id]
        if applicable:
            covered += 1
        old = previous.get(arb_id)
        if old is not None:
            for rule in applicable:
                index = rule['byte_index']
                if index >= len(old) or index >= len(payload):
                    continue
                kind = rule['kind']
                if kind == 'byte':
                    delta = abs(payload[index] - old[index])
                    if delta > rule['limit']:
                        entry['numeric_jump'] += 1
                else:
                    shift = 4 if kind == 'high_nibble' else 0
                    before, after = (old[index] >> shift) & 15, (payload[index] >> shift) & 15
                    if before != after and (after - before) % 16 != 1:
                        entry['counter'] += 1
        previous[arb_id] = payload
    table = [{'timestamp': min(end, start + (slot + 1) * bin_width), **bins[slot]}
             for slot in sorted(bins)]
    hours = (end - start) / 3600
    alert_bins = {method: sum(row[method] > 0 for row in table)
                  for method in ('counter', 'numeric_jump')}
    attack_bins = sum(row['attack_frames'] > 0 for row in table)
    detected = {method: sum(row[method] > 0 and row['attack_frames'] > 0 for row in table)
                for method in alert_bins}
    return {'source': str(path), 'sha256': file_hash(path), 'duration_seconds': end - start,
            'rows': len(rows), 'invalid_payloads': invalid, 'covered_frames': covered,
            'covered_fraction': covered / len(rows) if rows else 0,
            'alert_bins': alert_bins,
            'alert_bins_per_hour': {key: value / hours if hours else None
                                    for key, value in alert_bins.items()},
            'attack_bins': attack_bins, 'detected_attack_bins': detected,
            'attack_bin_recall': {key: value / attack_bins if attack_bins else None
                                  for key, value in detected.items()},
            'table': table}
