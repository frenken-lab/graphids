"""ID-first DBC annotation completion. Reference labels are evaluation only."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..decoding.dbc import load_dbc_fields
from ..decoding.signals import FieldCandidate, _bit_positions, discover_fields
from ..decoding.tokenization import tokenize
from ..preprocessing.capture import Capture
from ..preprocessing.traces import MessageTrace
from ..scoring.role_guided import ROLES, _score, _series
from ..scoring.dbc_labels import DBC_NAMES
from ..relationships.shared.signal_time import align, observations

FAMILY_BUDGET = 64
DETAIL_FAMILIES = 8
LAYOUT_BUDGET = 8


def _active(field: FieldCandidate, captures: list[Capture]) -> bool:
    for capture in captures:
        _, values = observations(capture, field)
        if len(values) < 30 or len(np.unique(values)) < 2:
            return False
    return True


def _value_equivalent(candidate: FieldCandidate, truth: FieldCandidate,
                      captures: list[Capture]) -> bool:
    """Fit an affine raw-value map on one route and check every route."""
    paired = []
    for capture in captures:
        try:
            values, valid, _, _ = align(capture, (candidate, truth), step=.05,
                                        max_age=.2, max_bins=8000)
        except ValueError:
            return False
        mask = valid.all(axis=1)
        if mask.sum() < 100:
            return False
        x, y = values[mask].T
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return False
        paired.append((x, y))
    x, y = paired[0]
    design = np.column_stack((x, np.ones_like(x)))
    gain, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    if abs(gain) < 1e-10:
        return False
    for x, y in paired:
        residual = np.mean((gain*x+offset-y)**2)
        baseline = np.var(y)
        if baseline < 1e-8 or 1-residual/baseline < .99:
            return False
    return True


def key(field: FieldCandidate) -> tuple:
    return (field.bus, field.arb_id.upper(), field.payload_length)


def layout_key(field: FieldCandidate) -> tuple:
    return (*key(field), field.start, field.width, field.endian, field.signed)


def interpretation_key(field: FieldCandidate) -> tuple:
    bits = _bit_positions(field.start, field.width, field.endian)
    return (bits, 'single_bit' if field.width == 1 else field.endian, field.signed)


def overlaps(left: FieldCandidate, right: FieldCandidate) -> bool:
    return key(left) == key(right) and bool(_bit_positions(left.start, left.width, left.endian) &
                                           _bit_positions(right.start, right.width, right.endian))


def rank_families(rows: list[tuple[FieldCandidate, float]]) -> list[dict]:
    """One best score per message family, independent of proposal count."""
    best = {}
    for field, score in rows:
        family = key(field)
        best[family] = max(best.get(family, -math.inf), score)
    ordered = sorted(best.items(), key=lambda item: (-item[1], repr(item[0])))
    return [{'rank': i, 'bus': family[0], 'arb_id': family[1],
             'payload_length': family[2], 'score': float(score)}
            for i, (family, score) in enumerate(ordered, 1)]


def enumerate_layouts(template: FieldCandidate, protocol_bits: set[int],
                      max_width: int = 32):
    """Adapt the shared exhaustive tokenizer to baseline field candidates."""
    trace = MessageTrace(template.arb_id, template.payload_length, template.bus,
                         (), ())
    for layout in tokenize(trace, method='exhaustive', max_width=max_width,
                           excluded_bits=protocol_bits).candidates():
        yield FieldCandidate(template.arb_id, layout.start, layout.width,
            layout.endian, layout.signed, 'unknown', 0., .5,
            template.reference_support, (('unknown', 1.),),
            ('contiguous_enumeration',), template.payload_length, bus=template.bus)


def compatible_layouts(rows: list[dict], limit: int = LAYOUT_BUDGET) -> list[dict]:
    # Each row is a one-field layout hypothesis. Overlap is allowed between
    # alternatives, never inside one alternative.
    return sorted(rows, key=lambda r: (-r['score'], repr(layout_key(r['field']))))[:limit]


def _protocol(registry, family: tuple) -> set[int]:
    return set().union(*(_bit_positions(f.start, f.width, f.endian)
        for f in registry.candidates if key(f) == family and
        f.role in ('counter', 'checksum'))) if registry.candidates else set()


def _reference_fields(dbc: Path, role: str, fitting: list[Capture],
                      development: list[Capture]) -> tuple[FieldCandidate, ...]:
    # Called only after ranking; no reference field is passed to the ranker.
    return tuple(f for f in load_dbc_fields(dbc)
                 if f.label_hypotheses[0][0].upper() in DBC_NAMES[role] and
                 _active(f, fitting) and any(_active(f, [route]) for route in development))


def attach_role_ranks(case: dict, fitting: list[Capture]) -> dict:
    """Rank all five role hypotheses per retained layout using fitting data."""
    for group in case['detailed_families']:
        family = (group['bus'], group['arb_id'], group['payload_length'])
        samples = _series(fitting, family)
        for row in group['candidate_layouts']:
            field = FieldCandidate(**row['layout'])
            ranked = sorted(((role, _score(field, samples, role)[0]) for role in ROLES),
                            key=lambda item: (-item[1], item[0]))
            row['role_hypotheses'] = [{'rank': rank, 'role': role,
                                       'score': score if math.isfinite(score) else None}
                                      for rank, (role, score) in enumerate(ranked, 1)]
    truths = {layout_key(FieldCandidate(**row)) for row in
              case['dbc_annotations']['reference_layouts']}
    role = case['hidden_role']
    ranks = [item['rank'] for group in case['detailed_families']
             for row in group['candidate_layouts']
             if layout_key(FieldCandidate(**row['layout'])) in truths
             for item in row['role_hypotheses'] if item['role'] == role]
    case['metrics']['role_rank'] = min(ranks) if ranks else None
    case['metrics'].pop('role_rank_reason', None)
    return case


def mine_case(vehicle: str, role: str, fitting: list[Capture],
              development: list[Capture], dbc: Path, registry=None) -> dict:
    if registry is None:
        registry = discover_fields(fitting, max_ids=FAMILY_BUDGET, max_layouts=LAYOUT_BUDGET)
    # The hidden DBC annotation is never used here. Other DBC roles are conflict anchors.
    anchors = [f for f in load_dbc_fields(dbc)
               if f.label_hypotheses[0][0].upper() not in DBC_NAMES[role]
               and any(f.label_hypotheses[0][0].upper() in names for names in DBC_NAMES.values())]
    by_family = defaultdict(list)
    for field in registry.candidates:
        if field.role == 'unknown' and field.signed is not None:
            by_family[key(field)].append(field)
    samples = {family: _series(fitting, family) for family in by_family}
    scored = []
    for family, fields in by_family.items():
        for field in fields:
            if any(overlaps(field, anchor) for anchor in anchors):
                continue
            score, evidence = _score(field, samples[family], role)
            if math.isfinite(score):
                scored.append((field, score))
    families = rank_families(scored)
    detailed = []
    for family_row in families[:DETAIL_FAMILIES]:
        family = (family_row['bus'], family_row['arb_id'], family_row['payload_length'])
        template = by_family[family][0]
        protocol = _protocol(registry, family)
        options = {}
        for field in [*by_family[family], *enumerate_layouts(template, protocol)]:
            if any(overlaps(field, anchor) for anchor in anchors):
                continue
            bits = _bit_positions(field.start, field.width, field.endian)
            if bits & protocol:
                continue
            identity = interpretation_key(field)
            if identity in options:
                continue
            score, evidence = _score(field, samples[family], role)
            if math.isfinite(score):
                options[identity] = {'field': field, 'score': float(score),
                                     'role_evidence': list(evidence)}
        ranked = sorted(options.values(), key=lambda r: (-r['score'], repr(layout_key(r['field']))))
        retained = compatible_layouts(ranked)
        detailed.append({**family_row, 'candidate_pool_size': len(ranked),
            'candidate_layouts': [{**{k:v for k,v in row.items() if k != 'field'},
                                   'layout': asdict(row['field']), 'rank': i}
                                  for i,row in enumerate(retained,1)],
            'null_layout': {'rank': None, 'reason': 'insufficient_or_ambiguous_evidence'},
            'ambiguity': len(ranked) > len(retained),
            'protocol_bits': sorted(protocol),
            'dbc_anchor_conflicts': [asdict(a) for a in anchors if key(a) == family]})
    # Evaluation begins only after all rankings are frozen in memory.
    references = _reference_fields(dbc, role, fitting, development)
    reference_families = {key(f) for f in references}
    id_rank = next((r['rank'] for r in families if
                    (r['bus'], r['arb_id'], r['payload_length']) in reference_families), None)
    exact_rank = None
    pool_coverage = False
    raw_agreement = False
    for group in detailed:
        family = (group['bus'], group['arb_id'], group['payload_length'])
        if family not in reference_families:
            continue
        # Coverage is computed from exhaustive enumeration, independently of retained rank.
        for truth in references:
            if key(truth) != family:
                continue
            bits = _bit_positions(truth.start, truth.width, truth.endian)
            pool_coverage |= (not bool(bits & set(group['protocol_bits'])) and
                              truth.width <= 32 and truth.mux_start is None and
                              not any(overlaps(truth, anchor) for anchor in anchors) and
                              math.isfinite(_score(replace(truth, confidence=.5),
                                                   samples[family], role)[0]))
            for row in group['candidate_layouts']:
                candidate = FieldCandidate(**row['layout'])
                if layout_key(candidate) == layout_key(truth):
                    exact_rank = min(exact_rank or 999, row['rank'])
                if not raw_agreement and _value_equivalent(candidate, truth, [*fitting, *development]):
                    raw_agreement = True
    return {'schema': 'dbc-label-completion-case-v2.1', 'vehicle': vehicle, 'hidden_role': role,
        'message_family_hypotheses': families, 'detailed_families': detailed,
        'dbc_annotations': {'hidden_reference_used_for_evaluation_only': True,
            'assessable': bool(references), 'reference_layouts': [asdict(f) for f in references]},
        'metrics': {'id_rank': id_rank, 'id_top_1': id_rank == 1,
            'id_top_5': id_rank is not None and id_rank <= 5,
            'id_top_8': id_rank is not None and id_rank <= 8,
            'exact_layout_rank_conditional_on_id_top_8': exact_rank if id_rank and id_rank <= 8 else None,
            'exact_layout_top_1': exact_rank == 1,
            'exact_layout_top_8': exact_rank is not None and exact_rank <= 8,
            'exact_layout_pool_coverage': pool_coverage,
            'held_out_raw_value_agreement': raw_agreement,
            'role_rank': None,
            'role_rank_reason': 'independent_role_calibration_not_available',
            'abstained': True},
        'review_ledger': {'proposed_name': role, 'status': 'role hypothesis',
                          'dbc_edit': False, 'physical_verification': False}}
