"""DBC-assisted anchors and source-gated ranking evidence."""
from __future__ import annotations

import json
import math
import numpy as np
from dataclasses import asdict
from pathlib import Path

from ..decoding.dbc import load_dbc_fields
from ..decoding.signals import FieldCandidate, _bit_positions
from ..label_completion.baseline import _active, key, layout_key
from .dbc_labels import DBC_NAMES
from ..label_completion.progressive_v6 import (
    RELATION_MIN_NET, RELATION_SOURCE_VEHICLES, RELATION_WEIGHT, anchor_series,
    candidate_features, decoded_values, field_from_row, fit_contrast, relation_net,
)
from .role_guided import _score, _series


def _family(row: dict) -> tuple:
    return row['bus'], row['arb_id'], row['payload_length']


def other_role_anchors(dbc: Path, hidden_role: str) -> tuple[FieldCandidate, ...]:
    """Use only known target fields for roles other than the hidden role."""
    return tuple(field for field in load_dbc_fields(dbc)
                 if field.label_hypotheses[0][0].upper() not in DBC_NAMES[hidden_role]
                 and any(field.label_hypotheses[0][0].upper() in names
                         for names in DBC_NAMES.values()))


def filter_known_role_anchors(layouts, family: dict,
                              anchors: tuple[FieldCandidate, ...]):
    """Exclude fields overlapping known other-role bits in the same family."""
    coordinates = (family['bus'], family['arb_id'].upper(), family['payload_length'])
    occupied = [bits for field in anchors if key(field) == coordinates
                for bits in (_bit_positions(field.start, field.width, field.endian),)]
    for layout in layouts:
        bits = _bit_positions(layout.start, layout.width, layout.endian)
        if not any(bits & anchor_bits for anchor_bits in occupied):
            yield layout


def target_relation_samples(fitting, anchor):
    return anchor_series(fitting, anchor) if anchor else None


def candidate_votes(row: dict, series, contrast, relation_samples):
    """Return source-trained and anchored adjustments for one candidate."""
    features = candidate_features(row, series, row['score'])
    contrast_valid = bool(np.isfinite(features).all())
    contrast_vote = contrast.adjustment(features) if contrast else 0.
    relation = relation_net(series, relation_samples) if relation_samples is not None else None
    relation_vote = RELATION_WEIGHT * relation if relation is not None else 0.
    return contrast_vote, relation_vote, relation, contrast_valid


def _row(field: FieldCandidate, score: float | None = None) -> dict:
    return {'bus': field.bus, 'arb_id': field.arb_id,
            'payload_length': field.payload_length, 'start': field.start,
            'width': field.width, 'endian': field.endian,
            'signed': field.signed, 'score': score}


def _source_examples(vehicle: str, role: str, fitting, case: dict, baseline: dict):
    """Annotated positives and high-ranking nonreference competitors."""
    positives = []
    negatives = []
    samples = {}
    references = [FieldCandidate(**row) for row in
                  baseline['dbc_annotations']['reference_layouts']]
    for field in references:
        family = key(field)
        samples.setdefault(family, _series(fitting, family))
        score, _ = _score(field_from_row(_row(field)), samples[family], role)
        if math.isfinite(score):
            positives.append(candidate_features(_row(field),
                decoded_values(_row(field), samples[family]), score))
    truth_families = {key(field) for field in references}
    competitors = []
    for family in case['families']:
        coordinates = _family(family)
        if coordinates in truth_families or not family['top_layouts']:
            continue
        top = family['top_layouts'][0]
        row = {**top, 'bus': family['bus'], 'arb_id': family['arb_id'],
               'payload_length': family['payload_length']}
        samples.setdefault(coordinates, _series(fitting, coordinates))
        features = candidate_features(row, decoded_values(row, samples[coordinates]),
                                      top['score'])
        if np_isfinite(features):
            competitors.append((family['family_rank'], coordinates, top['score'], features))
    negatives = [item[3] for item in competitors[:8]]
    return {'vehicle': vehicle, 'positives': positives, 'negatives': negatives,
            'competitors': competitors, 'truth_families': truth_families,
            'baseline_id_rank': case['reference_id_rank_evaluation_only']}


def np_isfinite(features) -> bool:
    return bool(all(math.isfinite(float(value)) for value in features))


def contrast_gate(target: str, role: str, loaded: dict, full_dir: Path,
                  baseline_dir: Path) -> tuple[object | None, dict]:
    source_rows = []
    for vehicle, data in loaded.items():
        if vehicle == target:
            continue
        fitting, _, _ = data
        case_path = full_dir / f'{vehicle.lower()}-{role}-prepared.json'
        base_path = baseline_dir / f'{vehicle.lower()}-{role}-baseline.json'
        case = json.loads(case_path.read_text())
        baseline = json.loads(base_path.read_text())
        if not case['assessable']:
            continue
        source_rows.append(_source_examples(vehicle, role, fitting, case, baseline))
    source_vehicles = [row['vehicle'] for row in source_rows
                       if row['positives'] and len(row['negatives']) >= 4]
    if len(source_vehicles) < 3:
        return None, {'status': 'rejected', 'reason': 'fewer_than_three_source_vehicles',
                      'source_vehicles': source_vehicles}
    checks = []
    for held in source_rows:
        training = [row for row in source_rows if row['vehicle'] != held['vehicle']]
        model = fit_contrast([p for row in training for p in row['positives']],
                             [n for row in training for n in row['negatives']])
        if model is None or not held['competitors'] or not held['positives']:
            return None, {'status': 'rejected', 'reason': 'source_fold_unavailable',
                          'source_vehicles': source_vehicles}
        # The held source's reference is used here only to assess the fold.
        # Its candidate family score is taken from its frozen v5 ID ranking.
        margins = [model.adjustment(p) for p in held['positives']]
        competitor_adjustments = [model.adjustment(item[3]) for item in held['competitors']]
        contrast = min(margins) - max(competitor_adjustments, default=0.)
        checks.append({'vehicle': held['vehicle'], 'positive_vs_hard_negative_margin': contrast,
                       'positive_adjustments': margins,
                       'max_competitor_adjustment': max(competitor_adjustments, default=0.)})
    # A source model must distinguish at least one annotated positive from
    # high-scoring competitors in every held vehicle. A failed gate has no vote.
    if any(row['positive_vs_hard_negative_margin'] <= 0 for row in checks):
        return None, {'status': 'rejected', 'reason': 'leave_one_vehicle_out_hard_negative_gate',
                      'source_vehicles': source_vehicles, 'checks': checks}
    model = fit_contrast([p for row in source_rows for p in row['positives']],
                         [n for row in source_rows for n in row['negatives']])
    return model, {'status': 'eligible', 'source_vehicles': source_vehicles,
                   'checks': checks, 'positive_count': sum(len(r['positives']) for r in source_rows),
                   'hard_competitor_count': sum(len(r['negatives']) for r in source_rows)}


def _active_anchor(fields: tuple[FieldCandidate, ...], role: str, fitting):
    options = [field for field in fields
               if field.label_hypotheses[0][0].upper() in DBC_NAMES[role]
               and _active(field, fitting)]
    return sorted(options, key=layout_key)[0] if options else None


def relation_gate(target: str, role: str, loaded: dict):
    if role not in ('vehicle_speed', 'engine_rpm'):
        return None, {'status': 'not_applicable'}
    other = 'engine_rpm' if role == 'vehicle_speed' else 'vehicle_speed'
    target_fitting, _, target_dbc = loaded[target]
    target_anchor = _active_anchor(load_dbc_fields(target_dbc), other, target_fitting)
    if target_anchor is None:
        return None, {'status': 'rejected', 'reason': 'target_anchor_unavailable'}
    supporters = []
    diagnostics = []
    for vehicle, (fitting, development, dbc) in loaded.items():
        if vehicle == target:
            continue
        source_fields = load_dbc_fields(dbc)
        candidate = _active_anchor(source_fields, role, fitting)
        anchor = _active_anchor(source_fields, other, fitting)
        if candidate is None or anchor is None:
            diagnostics.append({'vehicle': vehicle, 'status': 'unavailable'})
            continue
        route_nets = []
        for capture in [*fitting, *development]:
            family = key(candidate)
            sampled = _series([capture], family)
            route_nets.append(relation_net(decoded_values(_row(candidate), sampled),
                                           anchor_series([capture], anchor)))
        passed = all(net is not None and net >= RELATION_MIN_NET for net in route_nets)
        diagnostics.append({'vehicle': vehicle,
                            'status': 'supported' if passed else 'rejected',
                            'route_nets': route_nets})
        if passed:
            supporters.append(vehicle)
    if len(supporters) < RELATION_SOURCE_VEHICLES:
        return None, {'status': 'rejected', 'reason': 'source_replication_gate',
                      'supporters': supporters, 'diagnostics': diagnostics}
    return target_anchor, {'status': 'eligible', 'supporters': supporters,
                           'diagnostics': diagnostics,
                           'target_anchor': asdict(target_anchor)}
