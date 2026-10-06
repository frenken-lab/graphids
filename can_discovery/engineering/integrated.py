"""Experimental cross-vehicle frame matching and field-level evidence.

All fitting decisions are made before a target DBC is opened in source-only mode.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
from itertools import combinations

import numpy as np

from ..decoding.dbc import load_dbc_fields
from ..decoding.signals import FieldCandidate, FieldRegistry, _bit_positions, discover_fields
from ..preprocessing.capture import Capture
from ..relationships.physical.signal_graph import _edge
from ..relationships.shared.field_analysis import field_key
from ..relationships.shared.signal_time import align, observations


def family(field: FieldCandidate) -> tuple[str | None, str, int]:
    return field.bus, field.arb_id.upper(), field.payload_length


def observed_families(captures: list[Capture]) -> set[tuple[str | None, str, int]]:
    return set.intersection(*(set((frame.bus or capture.known_bus,
                                    frame.arb_id.upper(), len(frame.payload))
                                   for frame in capture.frames
                                   if frame.payload is not None and 1 <= len(frame.payload) <= 8)
                              for capture in captures)) if captures else set()


def _signature(field: FieldCandidate) -> tuple:
    return (field.start, field.width, field.endian, field.signed,
            field.mux_start, field.mux_value, field.label_hypotheses[0][0])


def _payload_profile(captures: list[Capture], arb_id: str, length: int,
                     bus: str | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    payloads = []
    for capture in captures:
        series = [np.frombuffer(frame.payload, dtype=np.uint8)
                  for frame in capture.frames if frame.payload is not None
                  and frame.arb_id.upper() == arb_id and len(frame.payload) == length
                  and (frame.bus or capture.known_bus) == bus]
        if len(series) < 40:
            return None
        bits = np.unpackbits(np.stack(series), axis=1).astype(float)
        payloads.append((bits.mean(axis=0), np.abs(np.diff(bits, axis=0)).mean(axis=0)))
    return (np.mean([item[0] for item in payloads], axis=0),
            np.mean([item[1] for item in payloads], axis=0))


def _profile_agreement(target: tuple[np.ndarray, np.ndarray] | None,
                       source: tuple[np.ndarray, np.ndarray] | None) -> tuple[bool, float | None]:
    if target is None or source is None:
        return False, None
    t_mean, t_flip = target
    s_mean, s_flip = source
    active_t = ((t_mean > .05) & (t_mean < .95)) | (t_flip > .01)
    active_s = ((s_mean > .05) & (s_mean < .95)) | (s_flip > .01)
    union = active_t | active_s
    if union.sum() < 2:
        return False, None
    overlap = (active_t & active_s).sum() / union.sum()
    distance = float(np.mean(np.abs(t_mean[union] - s_mean[union])) +
                     np.mean(np.abs(t_flip[union] - s_flip[union])))
    return bool(overlap >= .75 and distance <= .18), distance


def dbc_formats(path: str) -> dict[tuple[str, str], dict]:
    """Preserve physical scaling separately from raw bit coordinates."""
    import cantools

    database = cantools.database.load_file(path, strict=False)
    return {(format(message.frame_id, 'X'), signal.name):
            {'scale': signal.scale, 'offset': signal.offset, 'unit': signal.unit}
            for message in database.messages for signal in message.signals}


def decode_physical(capture: Capture, field: FieldCandidate,
                    value_format: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return timestamps and scaled values for a verified DBC interpretation."""
    if field.signed is None:
        raise ValueError('field signedness is unresolved')
    scale, offset = float(value_format['scale']), float(value_format['offset'])
    times, raw = observations(capture, field)
    return times, raw * scale + offset


def match_frames(target: list[Capture], sources: list[tuple[str, list[Capture], str]],
                 *, min_shared_ids: int = 2) -> tuple[tuple[FieldCandidate, ...], list[dict]]:
    """Rank same-ID/DLC references using other shared IDs; reject ambiguous layouts."""
    target_families = observed_families(target)
    target_ids = {arb for _, arb, _ in target_families}
    references = defaultdict(list)
    for vehicle, captures, dbc_path in sources:
        source_families = observed_families(captures)
        source_ids = {arb for _, arb, _ in source_families}
        overlap = len(target_ids & source_ids)
        source_profiles = {}
        formats = dbc_formats(dbc_path)
        for field in load_dbc_fields(dbc_path):
            matches = [key for key in source_families
                       if key[1:] == (field.arb_id.upper(), field.payload_length)]
            for source_family in matches:
                if source_family not in source_profiles:
                    source_profiles[source_family] = _payload_profile(
                        captures, source_family[1], source_family[2], source_family[0])
                references[source_family[1:]].append((vehicle, field, overlap,
                    captures[0].sha256, source_family[0], source_profiles[source_family],
                    formats.get((field.arb_id.upper(), field.label_hypotheses[0][0]))))
    accepted, decisions = [], []
    for bus, arb_id, length in sorted(target_families, key=repr):
        options = references.get((arb_id, length), [])
        if not options:
            continue
        ranked = sorted(options, key=lambda row: (-row[2], row[0], _signature(row[1])))
        best_overlap = ranked[0][2]
        best = [row for row in ranked if row[2] == best_overlap]
        target_profile = _payload_profile(target, arb_id, length, bus)
        agreements = [_profile_agreement(target_profile, row[5]) for row in best]
        payload_agreement = bool(agreements) and all(item[0] for item in agreements)
        signatures_by_vehicle = defaultdict(set)
        for vehicle, field, *_ in best:
            signatures_by_vehicle[vehicle].add(_signature(field))
        layouts = {tuple(sorted(signatures)) for signatures in signatures_by_vehicle.values()}
        unambiguous = (len(layouts) == 1 and best_overlap - 1 >= min_shared_ids
                       and payload_agreement)
        decisions.append({'bus': bus, 'arb_id': arb_id, 'payload_length': length,
            'status': 'anchor' if unambiguous else 'alternative',
            'shared_ids': best_overlap, 'sources': sorted(signatures_by_vehicle),
            'payload_agreement': payload_agreement,
            'payload_distance': min((distance for _, distance in agreements
                                     if distance is not None), default=None),
            'alternatives': [{'vehicle': vehicle, 'field': asdict(field),
                              'capture_hash': capture_hash, 'source_bus': source_bus,
                              'physical_format': value_format}
                             for vehicle, field, _, capture_hash, source_bus, _, value_format in ranked]})
        if not unambiguous:
            continue
        for field in sorted(next(iter(signatures_by_vehicle.values()))):
            exemplar = next(row[1] for row in best if _signature(row[1]) == field)
            accepted.append(replace(exemplar, bus=bus, role='known',
                evidence=('cross_vehicle_frame_match', *sorted(signatures_by_vehicle))))
    return tuple(accepted), decisions


def merge_registry(registry: FieldRegistry, anchors: tuple[FieldCandidate, ...]) -> FieldRegistry:
    """Keep alternatives in the pool and place nonoverlapping anchors first."""
    selected = []
    conflicts = list(registry.conflicts)
    for field in (*anchors, *registry.selected):
        bits = _bit_positions(field.start, field.width, field.endian)
        overlap = any(family(other) == family(field) and
            bits & _bit_positions(other.start, other.width, other.endian) and
            (field.mux_start is None or other.mux_start is None or
             (field.mux_start, field.mux_value) ==
             (other.mux_start, other.mux_value)) for other in selected)
        if overlap:
            conflicts.append({'bus': field.bus, 'arb_id': field.arb_id,
                              'payload_length': field.payload_length,
                              'reason': 'anchor_overlap' if field in anchors else 'token_overlap'})
            continue
        selected.append(field)
    keys = {field_key(field) for field in anchors}
    candidates = anchors + tuple(field for field in registry.candidates
                                  if field_key(field) not in keys)
    return FieldRegistry(candidates, tuple(selected), registry.tokenizations,
                         tuple(conflicts))


def discover_with_anchors(captures: list[Capture], anchors: tuple[FieldCandidate, ...],
                          *, max_ids: int = 64, max_layouts: int = 8) -> FieldRegistry:
    complete = set()
    for key in observed_families(captures):
        if any(field.mux_start is not None for field in anchors if family(field) == key):
            continue
        bits = set().union(*(_bit_positions(field.start, field.width, field.endian)
                            for field in anchors if family(field) == key))
        if len(bits) == key[2] * 8:
            complete.add(key)
    filtered = [replace(capture, frames=tuple(frame for frame in capture.frames
                if frame.payload is None or
                (frame.bus or capture.known_bus, frame.arb_id.upper(),
                 len(frame.payload)) not in complete)) for capture in captures]
    registry = discover_fields(filtered, max_ids=max_ids, max_layouts=max_layouts)
    return merge_registry(registry, anchors)


def field_relations(fitting: list[Capture], validation: list[Capture],
                    fields: tuple[FieldCandidate, ...], *, max_fields: int = 128,
                    max_pairs: int = 512,
                    priority_families: set[tuple[str | None, str, int]] | None = None
                    ) -> list[dict]:
    """Search all active fields, capped by support, and validate edges by route."""
    captures = [*fitting, *validation]
    counts_by_capture = []
    for capture in captures:
        counts = defaultdict(int)
        for frame in capture.frames:
            if frame.payload is not None:
                counts[(frame.bus or capture.known_bus, frame.arb_id.upper(),
                        len(frame.payload))] += 1
        counts_by_capture.append(counts)
    support = []
    for field in fields:
        counts = [item[family(field)] for item in counts_by_capture[:len(fitting)]]
        support.append(min(counts) if counts else 0)
    priority_families = priority_families or set()
    indices = sorted(range(len(fields)), key=lambda i:
                     (-(family(fields[i]) in priority_families),
                      -support[i], repr(field_key(fields[i]))))[:max_fields]
    results = []
    attempted = 0
    pair_captures = {}
    for left, right in combinations(indices, 2):
        if attempted >= max_pairs:
            break
        if family(fields[left]) == family(fields[right]):
            continue
        attempted += 1
        families = frozenset((family(fields[left]), family(fields[right])))
        if families not in pair_captures:
            pair_captures[families] = [replace(capture,
                frames=tuple(frame for frame in capture.frames if frame.payload is not None
                             and (frame.bus or capture.known_bus, frame.arb_id.upper(),
                                  len(frame.payload)) in families)) for capture in captures]
        edge = _edge(pair_captures[families], fields[left], fields[right], left, right)
        if edge.status == 'supported':
            results.append(asdict(edge))
    return results


def relation_groups(fields: tuple[FieldCandidate, ...], edges: list[dict]) -> list[dict]:
    """Only verified copies merge; correlated estimates stay separate."""
    parent = list(range(len(fields)))
    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for edge in edges:
        if edge['relation'] in ('exact_copy', 'affine_copy', 'synchronous_equivalent'):
            parent[root(edge['target'])] = root(edge['source'])
    groups = defaultdict(list)
    for index in range(len(fields)):
        groups[root(index)].append(index)
    return [{'members': members, 'observed_roots': [index for index in members
             if not any(edge['target'] == index and edge['source'] in members and
                        edge['relation'] in ('exact_copy', 'affine_copy') and
                        edge['lag_seconds'] for edge in edges)],
             'origin': 'unknown'} for members in groups.values()]


def derivative_relation(fitting: list[Capture], validation: list[Capture],
                        source: FieldCandidate, target: FieldCandidate,
                        *, operation: str) -> dict:
    """Fit an affine derivative/integral relation and check independent routes."""
    if operation not in ('derivative', 'integral'):
        raise ValueError(operation)
    model = None
    records = []
    for capture in (*fitting, *validation):
        try:
            values, valid, _, step = align(capture, (source, target), step=.1,
                                           max_age=.3, max_bins=8000)
        except ValueError:
            return {'status': 'unavailable', 'reason': 'invalid_time_grid'}
        mask = valid.all(axis=1)
        x, y = values.T
        transformed = np.gradient(x, step) if operation == 'derivative' else \
            np.cumsum(np.where(valid[:, 0], x, 0.)) * step
        if mask.sum() < 100 or min(np.std(transformed[mask]), np.std(y[mask])) < 1e-8:
            return {'status': 'unavailable', 'reason': 'insufficient_joint_activity'}
        if model is None:
            model = np.polyfit(transformed[mask], y[mask], 1)
        error = np.mean((y[mask] - np.polyval(model, transformed[mask])) ** 2)
        quality = 1 - error / max(np.var(y[mask]), 1e-10)
        shifted = np.roll(transformed, max(10, len(transformed)//3))
        control = 1 - np.mean((y[mask] - np.polyval(model, shifted[mask])) ** 2) / max(np.var(y[mask]), 1e-10)
        records.append({'samples': int(mask.sum()), 'quality': float(quality),
                        'shifted_quality': float(control)})
    supported = all(row['quality'] >= .8 and
                    row['quality'] - row['shifted_quality'] >= .05 for row in records)
    return {'status': 'supported' if supported else 'unsupported',
            'operation': operation, 'gain': float(model[0]),
            'offset': float(model[1]), 'routes': records}


def wheel_mean_relation(fitting: list[Capture], validation: list[Capture],
                        speed: FieldCandidate,
                        wheels: tuple[FieldCandidate, ...]) -> dict:
    """Check whether four wheel speeds jointly reproduce vehicle speed."""
    if len(wheels) != 4:
        raise ValueError('four wheel fields are required')
    model = None
    records = []
    for capture in (*fitting, *validation):
        try:
            values, valid, _, _ = align(capture, (speed, *wheels), step=.1,
                                        max_age=.3, max_bins=8000)
        except ValueError:
            return {'status': 'unavailable', 'reason': 'invalid_time_grid'}
        keep = valid.all(axis=1)
        if keep.sum() < 100:
            return {'status': 'unavailable', 'reason': 'insufficient_joint_samples'}
        y = values[:, 0]
        x = values[:, 1:].mean(axis=1)
        if min(np.std(x[keep]), np.std(y[keep])) < 1e-8:
            return {'status': 'unavailable', 'reason': 'insufficient_motion'}
        if model is None:
            model = np.polyfit(x[keep], y[keep], 1)
        quality = 1 - np.mean((y[keep] - np.polyval(model, x[keep])) ** 2) / np.var(y[keep])
        shifted = np.roll(x, max(10, len(x)//3))
        control = 1 - np.mean((y[keep] - np.polyval(model, shifted[keep])) ** 2) / np.var(y[keep])
        records.append({'samples': int(keep.sum()), 'quality': float(quality),
                        'shifted_quality': float(control)})
    supported = model[0] > 0 and all(item['quality'] >= .8 and
        item['quality']-item['shifted_quality'] >= .05 for item in records)
    return {'status': 'supported' if supported else 'unsupported',
            'operation': 'four_wheel_mean', 'gain': float(model[0]),
            'offset': float(model[1]), 'routes': records}
