"""Experimental, observational field and physical-role graph.

This module deliberately does not alter version-11 profiles.  Edges describe
replicated observations, not intervention-level or ECU-origin causality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from ...decoding.field_values import extract
from ...decoding.signals import (
    FieldCandidate,
    FieldRegistry,
    _bit_positions,
    _infer_signed,
)
from ...preprocessing.capture import Capture
from ...preprocessing.tokenization import MessageTrace, tokenize
from ..shared.field_analysis import field_key
from ..shared.signal_time import align
from .candidate_graph import layout_memberships
from .observational_edges import GRAPH_EDGES, evaluate_graph_edge
from .role_types import ROLES, PhysicalDiscovery


@dataclass(frozen=True, slots=True)
class FieldEdge:
    source: int
    target: int
    relation: str
    status: str
    lag_seconds: float | None
    value_error: float | None
    change_agreement: float | None
    support: tuple[int, ...]
    reason: str = ''


def _key(field: FieldCandidate) -> tuple:
    return (field.bus, field.arb_id, field.payload_length, field.start,
            field.width, field.endian, field.signed, field.mux_start,
            field.mux_value)


def _active_values(capture: Capture, field: FieldCandidate) -> np.ndarray:
    values = []
    for frame in capture.frames:
        if (frame.arb_id != field.arb_id or frame.payload is None or
            len(frame.payload) != field.payload_length or
            (frame.bus or capture.known_bus) != field.bus):
            continue
        if (field.mux_start is not None and
            frame.payload[field.mux_start // 8] != field.mux_value):
            continue
        values.append(extract(frame.payload, field.start, field.width,
                              field.endian, field.signed))
    return np.asarray(values, dtype=float)


def _paired(capture: Capture, left: FieldCandidate, right: FieldCandidate,
            lag: int, *, step: float = .02) -> tuple[np.ndarray, np.ndarray, float]:
    values, valid, _, actual_step = align(capture, (left, right), step=step,
                                          max_age=.1, max_bins=6000)
    if lag > 0:
        values, valid = np.column_stack((values[:-lag, 0], values[lag:, 1])), \
                        np.column_stack((valid[:-lag, 0], valid[lag:, 1]))
    elif lag < 0:
        shift = -lag
        values, valid = np.column_stack((values[shift:, 0], values[:-shift, 1])), \
                        np.column_stack((valid[shift:, 0], valid[:-shift, 1]))
    keep = valid.all(axis=1)
    return values[keep, 0], values[keep, 1], actual_step


def _edge(captures: list[Capture], left: FieldCandidate,
          right: FieldCandidate, source: int, target: int) -> FieldEdge:
    if (left.bus, left.arb_id, left.payload_length) == (right.bus, right.arb_id,
                                                       right.payload_length):
        return FieldEdge(source, target, 'unknown', 'unavailable', None, None,
                         None, (), 'same_message_family')
    if any(min(len(_active_values(c, f)) for c in captures) < 40 for f in (left, right)):
        return FieldEdge(source, target, 'unknown', 'unavailable', None, None,
                         None, (), 'insufficient_frames')
    # Fit the affine map and lag on the first independent route only.
    trials = []
    for lag in range(-5, 6):
        x, y, actual_step = _paired(captures[0], left, right, lag)
        if len(x) < 150 or min(np.std(x), np.std(y)) < 1e-8:
            continue
        slope, offset = np.polyfit(x, y, 1)
        scale = max(float(np.ptp(y)), 1.)
        error = float(np.quantile(np.abs(y - (slope*x + offset)), .99) / scale)
        changes = np.diff(x), np.diff(y)
        change = (float(np.corrcoef(*changes)[0, 1])
                  if min(np.std(v) for v in changes) > 1e-8 else 0.)
        trials.append((error, -change, abs(lag), lag, float(slope), float(offset),
                       actual_step))
    if not trials:
        return FieldEdge(source, target, 'unknown', 'unavailable', None, None,
                         None, (), 'insufficient_activity')
    error, neg_change, _, lag, slope, offset, actual_step = min(trials)
    support = []
    worst_error, weakest_change = error, -neg_change
    control_separation = []
    for capture in captures:
        x, y, _ = _paired(capture, left, right, lag)
        if len(x) < 150 or min(np.std(x), np.std(y)) < 1e-8:
            return FieldEdge(source, target, 'unknown', 'unavailable', None,
                             None, None, tuple(support), 'insufficient_joint_activity')
        scale = max(float(np.ptp(y)), 1.)
        actual_error = float(np.quantile(np.abs(y - (slope*x + offset)), .99) / scale)
        worst_error = max(worst_error, actual_error)
        dx, dy = np.diff(x), np.diff(y)
        change = (float(np.corrcoef(dx, dy)[0, 1])
                  if min(np.std(dx), np.std(dy)) > 1e-8 else 0.)
        weakest_change = min(weakest_change, change)
        control_x, control_y, _ = _paired(capture, left, right, lag+20)
        if len(control_x) >= 150 and np.ptp(control_y) > 0:
            control_error = float(np.quantile(np.abs(
                control_y-(slope*control_x+offset)), .99) /
                max(float(np.ptp(control_y)), 1.))
            control_separation.append(control_error - actual_error)
        support.append(len(x))
    if control_separation and min(control_separation) <= .005:
        return FieldEdge(source, target, 'unknown', 'unsupported',
                         abs(lag)*actual_step, worst_error, weakest_change,
                         tuple(support), 'shifted_control_not_separated')
    if worst_error <= .01 and weakest_change >= .9 and slope > 0:
        relation = ('synchronous_equivalent' if lag == 0 else
                    'exact_copy' if abs(slope-1) <= .01 and abs(offset) <= 1
                    else 'affine_copy')
    elif weakest_change >= .6 and worst_error <= .1:
        relation = 'transformed_estimate'
    else:
        relation = 'related_or_distinct'
    # Zero-lag observations do not establish which field came first.
    if lag < 0:
        source, target = target, source
    return FieldEdge(source, target, relation, 'supported' if relation !=
                     'related_or_distinct' else 'unsupported', abs(lag)*actual_step,
                     worst_error, weakest_change, tuple(support))


def _layout_records(registry: FieldRegistry) -> list[dict]:
    records = []
    for result in registry.tokenizations:
        alternatives = result.alternatives
        records.append({'bus': result.trace.bus, 'arb_id': result.trace.arb_id,
                        'payload_length': result.trace.payload_length,
                        'support': result.trace.support,
                        'abstentions': result.abstention_reasons,
                        'alternatives': [{'rank': rank, 'cost': alt.cost,
                            'spans': [asdict(span) for span in alt.spans],
                            'unresolved_bits': alt.unresolved_bits}
                            for rank, alt in enumerate(alternatives, 1)]})
    return records


def conditional_layouts(registry: FieldRegistry, captures: list[Capture],
                        *, max_layouts: int = 8, min_mode_frames: int = 40) -> list[dict]:
    """Infer conservative byte-wide modes; unresolved families simply abstain."""
    out = []
    for result in registry.tokenizations:
        trace = result.trace
        family = [[frame.payload for frame in capture.frames
                   if frame.payload is not None and frame.arb_id == trace.arb_id
                   and len(frame.payload) == trace.payload_length and
                   (frame.bus or capture.known_bus) == trace.bus]
                  for capture in captures]
        for byte in range(trace.payload_length):
            modes = set().union(*(set(p[byte] for p in trip) for trip in family))
            if not 2 <= len(modes) <= 4:
                continue
            if any(any(sum(p[byte] == mode for p in trip) < min_mode_frames
                       for mode in modes) for trip in family):
                continue
            # A selector should switch much less frequently than ordinary data.
            if any(np.mean(np.diff([p[byte] for p in trip]) != 0) > .2
                   for trip in family):
                continue
            mode_records = []
            for mode in sorted(modes):
                samples = tuple(tuple(p for p in trip if p[byte] == mode)[:1000]
                                for trip in family)
                conditional = tokenize(MessageTrace(trace.arb_id, trace.payload_length,
                    trace.bus, samples, trace.source_hashes), max_layouts=max_layouts)
                mode_records.append({'value': mode, 'support': tuple(map(len, samples)),
                    'alternatives': [{'cost': alt.cost,
                        'spans': [asdict(span) for span in alt.spans
                                  if not set(span.physical_bits) &
                                  set(range(byte*8, byte*8+8))]}
                        for alt in conditional.alternatives]})
            out.append({'bus': trace.bus, 'arb_id': trace.arb_id,
                        'payload_length': trace.payload_length,
                        'selector_byte': byte, 'modes': mode_records,
                        'status': 'candidate_only'})
            break
    return out


def conditional_field_candidates(layouts: list[dict],
                                 captures: list[Capture]) -> tuple[FieldCandidate, ...]:
    """Turn supported mode layouts into conditional field hypotheses."""
    fields = []
    seen = set()
    for record in layouts:
        selector = record['selector_byte']
        for mode in record['modes']:
            samples = [[frame.payload for frame in capture.frames
                        if frame.payload is not None and
                        frame.arb_id == record['arb_id'] and
                        len(frame.payload) == record['payload_length'] and
                        (frame.bus or capture.known_bus) == record['bus'] and
                        frame.payload[selector] == mode['value']]
                       for capture in captures]
            for alternative in mode['alternatives']:
                for span in alternative['spans']:
                    endian = span['byte_order']
                    if endian == 'unresolved':
                        continue
                    bits = span['physical_bits']
                    if endian == 'little':
                        mapped = {(bit//8)*8 + 7-bit%8 for bit in bits}
                        if max(mapped)-min(mapped)+1 != len(mapped):
                            continue
                        start = min(mapped)
                    else:
                        start = min(bits)
                    key = (record['bus'], record['arb_id'], record['payload_length'],
                           selector, mode['value'], start, len(bits), endian)
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        signed = _infer_signed(samples, start, len(bits), endian)
                    except (ValueError, OverflowError):
                        signed = None
                    fields.append(FieldCandidate(record['arb_id'], start, len(bits),
                        endian, signed, 'unknown', 0., 1.,
                        tuple(map(len, samples)), (('unknown', 1.),),
                        ('conditional_layout',), record['payload_length'],
                        selector*8, mode['value'], record['bus']))
    return tuple(fields)


def _field_layouts(fields: list[FieldCandidate], registry: FieldRegistry,
                   modes: list[dict]) -> list[list[str]]:
    ordinary = layout_memberships(tuple(fields), registry.tokenizations)
    output = []
    for field in fields:
        if field.mux_start is None:
            output.append(list(ordinary.get(field_key(field), ())))
            continue
        candidates = []
        for record in modes:
            if (record['bus'], record['arb_id'], record['payload_length'],
                record['selector_byte']*8) != (field.bus, field.arb_id,
                    field.payload_length, field.mux_start):
                continue
            for mode in record['modes']:
                if mode['value'] != field.mux_value:
                    continue
                for rank, alternative in enumerate(mode['alternatives']):
                    for span in alternative['spans']:
                        if span['byte_order'] != field.endian or \
                                len(span['physical_bits']) != field.width:
                            continue
                        bits = span['physical_bits']
                        matches = (min(bits) == field.start if field.endian == 'big'
                                   else min((bit//8)*8+7-bit%8 for bit in bits) ==
                                   field.start)
                        if matches:
                            candidates.append(f'{field.arb_id}:mux{record["selector_byte"]}'
                                              f'={mode["value"]}:layout{rank}')
        output.append(candidates)
    return output


def build_signal_graph(captures: list[Capture], registry: FieldRegistry,
                       discovery: PhysicalDiscovery, *, max_per_role: int = 6,
                       modes: list[dict] | None = None) -> dict:
    """Build a bounded graph using fitting and development-validation routes."""
    if len(captures) < 2 or len({c.sha256 for c in captures}) != len(captures):
        raise ValueError('independent fitting and validation captures are required')
    selected = {}
    for role in ROLES:
        ranked = sorted((h for h in discovery.hypotheses if h.role == role),
                        key=lambda h: -h.unary_score)
        for hypothesis in ranked[:max_per_role]:
            selected.setdefault(_key(hypothesis.candidate), hypothesis.candidate)
    fields = list(selected.values())
    modes = modes if modes is not None else conditional_layouts(registry, captures)
    field_layouts = _field_layouts(fields, registry, modes)
    index = {_key(field): i for i, field in enumerate(fields)}
    memberships = {role: [] for role in ROLES}
    unary = {role: {} for role in ROLES}
    for hypothesis in discovery.hypotheses:
        key = _key(hypothesis.candidate)
        if key in index:
            memberships[hypothesis.role].append(index[key])
            unary[hypothesis.role][index[key]] = max(
                unary[hypothesis.role].get(index[key], 0.), hypothesis.unary_score)
    edges = []
    compared = set()
    for role in ROLES:
        members = sorted(set(memberships[role]))
        for a, b in combinations(members, 2):
            if (a,b) in compared:
                continue
            compared.add((a,b))
            edges.append(_edge(captures, fields[a], fields[b], a, b))
    # Copies form a representation family. Estimates remain adjacent evidence,
    # since wheel speeds and vehicle speed can be strongly related but distinct.
    parent = list(range(len(fields)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for edge in edges:
        if edge.status == 'supported' and edge.relation in (
            'exact_copy', 'affine_copy', 'synchronous_equivalent'):
            parent[root(edge.target)] = root(edge.source)
    groups = []
    for role in ROLES:
        buckets = {}
        for member in sorted(set(memberships[role])):
            buckets.setdefault(root(member), []).append(member)
        for members in buckets.values():
            incoming = {e.target for e in edges if e.status == 'supported'
                        and e.relation in ('exact_copy','affine_copy') and
                        e.lag_seconds and e.source in members and e.target in members}
            groups.append({'role': role, 'members': members,
                           'observed_roots': [i for i in members if i not in incoming],
                           'score': max(unary[role][i] for i in members) +
                                    .02*min(2, len(members)-1),
                           'origin': 'unknown'})
    groups.sort(key=lambda group: (ROLES.index(group['role']), -group['score'],
                                   group['members']))
    roots_by_role = {role: list(dict.fromkeys(root for group in groups
                     if group['role'] == role for root in group['observed_roots']))
                     for role in ROLES}
    role_edges = []
    for name, (source_role, target_role, _) in GRAPH_EDGES.items():
        left = roots_by_role[source_role][:max_per_role]
        right = roots_by_role[target_role][:max_per_role]
        candidates = []
        for a in left:
            for b in right:
                first, second = fields[a], fields[b]
                if (first.bus, first.arb_id, first.payload_length,
                    first.mux_start, first.mux_value) == (
                    second.bus, second.arb_id, second.payload_length,
                    second.mux_start, second.mux_value):
                    if (not set(field_layouts[a]) & set(field_layouts[b]) or
                        _bit_positions(first.start, first.width, first.endian) &
                        _bit_positions(second.start, second.width, second.endian)):
                        continue
                evidence = evaluate_graph_edge(captures, fields[a], fields[b], name)
                if evidence.status == 'supported':
                    candidates.append((evidence.gain, a, b, evidence))
        if candidates:
            _, a, b, evidence = max(candidates, key=lambda x: x[0])
            role_edges.append({'name': name, 'source': a, 'target': b,
                               **asdict(evidence)})
        else:
            role_edges.append({'name': name, 'status': 'unsupported'})
    return {'version': 1, 'capture_hashes': [c.sha256 for c in captures],
            'layout_candidates': _layout_records(registry),
            'conditional_layouts': modes,
            'layout_conflicts': list(registry.conflicts),
            'fields': [asdict(field) for field in fields],
            'field_layouts': field_layouts,
            'field_edges': [asdict(edge) for edge in edges],
            'role_groups': groups, 'role_edges': role_edges,
            'null_roles': [role for role in ROLES if not memberships[role]],
            'claim': 'observed_root_only'}
