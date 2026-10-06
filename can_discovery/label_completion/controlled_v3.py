"""Controlled ID expansion and source-gated relational reranking.

Consumes frozen v2.1 cases. Hidden annotations are read only for diagnostics.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..decoding.dbc import load_dbc_fields
from ..decoding.signals import FieldCandidate, _bit_positions
from ..preprocessing.capture import Capture
from ..relationships.shared.signal_time import align
from .pipeline import (DBC_NAMES, ROLES, _active, _protocol, _score, _series,
                       _value_equivalent, compatible_layouts, enumerate_layouts,
                       interpretation_key, key, layout_key, overlaps)

SCHEMA = 'dbc-label-completion-controlled-v3'
BASE_DETAIL = 8
EXPANDED_DETAIL = 16
CUTOFF_GAP = .10
RELATION_WEIGHT = .15
SOURCE_VEHICLES_REQUIRED = 2
SOURCE_SPLIT_MIN_NET = .04


def adaptive_limit(families: list[dict]) -> int:
    """Allocate eight more IDs when ranks 8–16 are poorly separated."""
    if len(families) < EXPANDED_DETAIL:
        return min(BASE_DETAIL, len(families))
    gap = families[BASE_DETAIL-1]['score'] - families[EXPANDED_DETAIL-1]['score']
    return EXPANDED_DETAIL if gap <= CUTOFF_GAP else BASE_DETAIL


def relation_route(capture: Capture, speed: FieldCandidate,
                   rpm: FieldCandidate) -> dict:
    """Positive synchronized change correlation beyond a shifted control."""
    try:
        values, valid, _, _ = align(capture, (speed, rpm), step=.1,
                                    max_age=.3, max_bins=4096)
    except ValueError:
        return {'status': 'unavailable', 'reason': 'invalid_time_grid'}
    paired = values[valid.all(axis=1)]
    if len(paired) < 120:
        return {'status': 'unavailable', 'reason': 'insufficient_joint_samples'}
    x, y = np.diff(paired[:, 0]), np.diff(paired[:, 1])
    if min(np.std(x), np.std(y)) < 1e-8:
        return {'status': 'unavailable', 'reason': 'insufficient_change'}
    actual = float(np.corrcoef(x, y)[0, 1])
    control = float(np.corrcoef(x, np.roll(y, max(10, len(y)//3)))[0, 1])
    return {'status': 'measured', 'actual': actual, 'shifted_control': control,
            'net': max(0., actual-control), 'samples': int(len(paired))}


def source_pair(fields: tuple[FieldCandidate, ...], fitting: list[Capture]
                ) -> tuple[FieldCandidate, FieldCandidate] | None:
    selected = []
    for role in ('vehicle_speed', 'engine_rpm'):
        options = [f for f in fields if
                   f.label_hypotheses[0][0].upper() in DBC_NAMES[role]
                   and _active(f, fitting)]
        if not options:
            return None
        selected.append(sorted(options, key=layout_key)[0])
    return selected[0], selected[1]


def source_gate(target: str, sources: dict[str, tuple[list[Capture],
                list[Capture], Path]]) -> dict:
    vehicles = []
    for vehicle, (fitting, development, dbc) in sorted(sources.items()):
        if vehicle == target:
            continue
        pair = source_pair(load_dbc_fields(dbc), fitting)
        if pair is None:
            vehicles.append({'vehicle': vehicle, 'status': 'unavailable',
                             'reason': 'speed_or_rpm_anchor_unavailable'})
            continue
        splits = {name: [relation_route(c, *pair) for c in captures]
                  for name, captures in (('fitting', fitting),
                                         ('development', development))}
        means = {name: (sum(r['net'] for r in rows)/len(rows)
                        if all(r['status'] == 'measured' for r in rows) else None)
                 for name, rows in splits.items()}
        supported = all(x is not None and x >= SOURCE_SPLIT_MIN_NET
                        for x in means.values())
        vehicles.append({'vehicle': vehicle,
                         'status': 'supported' if supported else 'rejected',
                         'reference_layouts': [asdict(f) for f in pair],
                         'splits': splits, 'mean_net_by_split': means})
    supporters = [r['vehicle'] for r in vehicles if r['status'] == 'supported']
    return {'relation': 'speed_rpm_positive_change_vs_shifted_control',
            'status': 'eligible' if len(supporters) >= SOURCE_VEHICLES_REQUIRED
                      else 'rejected',
            'supporting_vehicles': supporters,
            'required_vehicles': SOURCE_VEHICLES_REQUIRED,
            'source_split_min_net': SOURCE_SPLIT_MIN_NET,
            'vehicles': vehicles}


def target_anchor(role: str, fitting: list[Capture], dbc: Path
                  ) -> FieldCandidate | None:
    other = {'engine_rpm': 'vehicle_speed',
             'vehicle_speed': 'engine_rpm'}.get(role)
    if other is None:
        return None
    options = [f for f in load_dbc_fields(dbc)
               if f.label_hypotheses[0][0].upper() in DBC_NAMES[other]
               and _active(f, fitting)]
    return sorted(options, key=layout_key)[0] if options else None


def relation_score(fitting: list[Capture], candidate: FieldCandidate,
                   anchor: FieldCandidate, role: str) -> float:
    speed, rpm = ((anchor, candidate) if role == 'engine_rpm'
                  else (candidate, anchor))
    rows = [relation_route(c, speed, rpm) for c in fitting]
    if not all(r['status'] == 'measured' for r in rows):
        return 0.
    return float(np.mean([r['net'] for r in rows]))


def family_key(row: dict) -> tuple:
    return row['bus'], row['arb_id'], row['payload_length']


def detail_family(family_row: dict, role: str, fitting: list[Capture],
                  registry, anchors: list[FieldCandidate]) -> dict:
    """Apply v2.1 layout search to one newly allocated family."""
    family = family_key(family_row)
    proposals = [f for f in registry.candidates if key(f) == family
                 and f.role == 'unknown' and f.signed is not None]
    protocol = _protocol(registry, family)
    options = {}
    if proposals:
        samples = _series(fitting, family)
        for field in [*proposals, *enumerate_layouts(proposals[0], protocol)]:
            if any(overlaps(field, anchor) for anchor in anchors):
                continue
            if _bit_positions(field.start, field.width, field.endian) & protocol:
                continue
            identity = interpretation_key(field)
            if identity in options:
                continue
            score, evidence = _score(field, samples, role)
            if math.isfinite(score):
                options[identity] = {'field': field, 'score': float(score),
                                     'role_evidence': list(evidence)}
    retained = compatible_layouts(list(options.values()))
    rows = []
    for rank, option in enumerate(retained, 1):
        field = option['field']
        role_scores = sorted(((name, _score(field, samples, name)[0])
                              for name in ROLES),
                             key=lambda item: (-item[1], item[0]))
        rows.append({'rank': rank, 'layout': asdict(field),
                     'score': option['score'],
                     'role_evidence': option['role_evidence'],
                     'role_hypotheses': [{'rank': i, 'role': name,
                        'score': score if math.isfinite(score) else None}
                        for i, (name, score) in enumerate(role_scores, 1)]})
    return {**family_row, 'candidate_pool_size': len(options),
            'candidate_layouts': rows,
            'null_layout': {'rank': None, 'reason': 'insufficient_or_ambiguous_evidence'},
            'ambiguity': len(options) > len(rows), 'protocol_bits': sorted(protocol),
            'dbc_anchor_conflicts': [asdict(a) for a in anchors if key(a) == family]}


def evaluate(groups: list[dict], ranked_families: list[dict],
             reference: list[FieldCandidate], fitting: list[Capture],
             development: list[Capture], role: str,
             raw_cache: dict | None = None) -> dict:
    raw_cache = {} if raw_cache is None else raw_cache
    truth_families = {key(f) for f in reference}
    id_rank = next((i for i, row in enumerate(ranked_families, 1)
                    if family_key(row) in truth_families), None)
    exact_rank = None
    raw_agreement = False
    role_rank = None
    for group in groups:
        if family_key(group) not in truth_families:
            continue
        for row in group['candidate_layouts']:
            field = FieldCandidate(**row['layout'])
            for truth in reference:
                if layout_key(field) == layout_key(truth):
                    exact_rank = min(exact_rank or 999, row['rank'])
                    role_rank = min(role_rank or 999, next(item['rank']
                        for item in row['role_hypotheses'] if item['role'] == role))
                pair = (layout_key(field), layout_key(truth))
                if not raw_agreement and pair not in raw_cache:
                    raw_cache[pair] = _value_equivalent(field, truth,
                                                         [*fitting, *development])
                if raw_cache.get(pair, False):
                    raw_agreement = True
    return {'id_rank': id_rank, 'id_top_1': id_rank == 1,
            'id_top_5': id_rank is not None and id_rank <= 5,
            'id_top_8': id_rank is not None and id_rank <= 8,
            'exact_layout_rank': exact_rank,
            'exact_layout_top_1': exact_rank == 1,
            'exact_layout_top_8': exact_rank is not None and exact_rank <= 8,
            'role_rank_conditional_on_exact_layout': role_rank,
            'held_out_affine_raw_agreement': raw_agreement, 'abstained': True}


def rerank(groups: list[dict], family_rows: list[dict], fitting: list[Capture],
           anchor: FieldCandidate | None, role: str, eligible: bool
           ) -> tuple[list[dict], list[dict]]:
    if not eligible or anchor is None:
        return groups, family_rows
    rescored = []
    for group in groups:
        changed = {**group, 'candidate_layouts': []}
        for row in group['candidate_layouts']:
            relation = relation_score(fitting, FieldCandidate(**row['layout']),
                                      anchor, role)
            changed['candidate_layouts'].append({
                **row, 'relation_score': relation,
                'joint_score': row['score'] + RELATION_WEIGHT*relation})
        changed['candidate_layouts'].sort(key=lambda r: (-r['joint_score'],
            repr(layout_key(FieldCandidate(**r['layout'])))))
        for rank, row in enumerate(changed['candidate_layouts'], 1):
            row['rank'] = rank
        changed['relation_score'] = max(
            (r['relation_score'] for r in changed['candidate_layouts']), default=0.)
        changed['joint_score'] = group['score'] + RELATION_WEIGHT*changed['relation_score']
        rescored.append(changed)
    by_family = {family_key(group): group for group in rescored}
    ranked = [{**row, 'joint_score': by_family[family_key(row)]['joint_score']}
              for row in family_rows if family_key(row) in by_family]
    ranked.sort(key=lambda row: (-row['joint_score'], repr(family_key(row))))
    return rescored, ranked


def run_case(base: dict, fitting: list[Capture], development: list[Capture],
             dbc: Path, registry, gate: dict) -> dict:
    """Compare equal detailed-search budgets before consulting hidden truth."""
    role = base['hidden_role']
    families = base['message_family_hypotheses']
    budget = adaptive_limit(families)
    base_groups = base['detailed_families']
    details = {family_key(row): row for row in base_groups}
    annotations = load_dbc_fields(dbc)
    anchors = [f for f in annotations
               if f.label_hypotheses[0][0].upper() not in DBC_NAMES[role]
               and any(f.label_hypotheses[0][0].upper() in names
                       for names in DBC_NAMES.values())]
    anchor = target_anchor(role, fitting, dbc) if gate['status'] == 'eligible' else None
    eligible = gate['status'] == 'eligible' and anchor is not None

    # Only tokenizer proposals are used to choose which extra IDs to open.
    by_family = defaultdict(list)
    if eligible:
        for field in registry.candidates:
            if field.role == 'unknown' and field.signed is not None and not any(
                    overlaps(field, other) for other in anchors):
                if len(by_family[key(field)]) < 4:
                    by_family[key(field)].append(field)
    relation_scores = {}
    for row in families:
        candidates = by_family.get(family_key(row), ())
        relation_scores[family_key(row)] = (max(
            (relation_score(fitting, f, anchor, role) for f in candidates),
            default=0.) if eligible else 0.)

    unary_selection = families[:budget]
    feedback_selection = list(families[:BASE_DETAIL])
    if budget > BASE_DETAIL:
        extras = sorted(families[BASE_DETAIL:], key=lambda row:
            (-row['score'] - RELATION_WEIGHT*relation_scores[family_key(row)],
             repr(family_key(row))))[:budget-BASE_DETAIL]
        feedback_selection.extend(extras)
    for row in [*unary_selection, *feedback_selection]:
        family = family_key(row)
        if family not in details:
            details[family] = detail_family(row, role, fitting, registry, anchors)
    unary_groups = [details[family_key(row)] for row in unary_selection]
    feedback_groups = [details[family_key(row)] for row in feedback_selection]
    fixed_groups, fixed_ranked = rerank(unary_groups, unary_selection, fitting,
                                         anchor, role, eligible)
    feedback_groups, feedback_ranked = rerank(feedback_groups,
        feedback_selection, fitting, anchor, role, eligible)

    # All selection and reranking above is complete before hidden reference use.
    references = [FieldCandidate(**row) for row in
                  base['dbc_annotations']['reference_layouts']]
    truth_families = {key(f) for f in references}
    oracle_groups = []
    for family in sorted(truth_families, key=repr):
        if family not in details:
            row = next((r for r in families if family_key(r) == family), None)
            if row is not None:
                details[family] = detail_family(row, role, fitting, registry, anchors)
        if family in details:
            oracle_groups.append(details[family])

    raw_cache = {}
    baseline = evaluate(base_groups, families, references, fitting, development,
                        role, raw_cache)
    expanded = evaluate(unary_groups, unary_selection, references,
                        fitting, development, role, raw_cache)
    fixed = evaluate(fixed_groups, fixed_ranked, references,
                     fitting, development, role, raw_cache)
    feedback = evaluate(feedback_groups, feedback_ranked, references,
                        fitting, development, role, raw_cache)
    oracle = evaluate(oracle_groups, families, references,
                      fitting, development, role, raw_cache)
    detailed_before = any(family_key(g) in truth_families for g in base_groups)
    oracle_generated = False
    for group in oracle_groups:
        family = family_key(group)
        samples = _series(fitting, family)
        for truth in references:
            if key(truth) != family:
                continue
            oracle_generated |= (truth.width <= 32 and truth.mux_start is None and
                not bool(_bit_positions(truth.start, truth.width, truth.endian) &
                         set(group['protocol_bits'])) and
                not any(overlaps(truth, a) for a in anchors) and
                math.isfinite(_score(truth, samples, role)[0]))
    trace = {'initial_pool': baseline['id_rank'] is not None,
             'initial_id_rank': baseline['id_rank'],
             'baseline_detailed': detailed_before,
             'baseline_exact_generated': (
                 base['metrics']['exact_layout_pool_coverage'] if detailed_before else None),
             'baseline_exact_retained': baseline['exact_layout_rank'] is not None,
             'baseline_role_rank': baseline['role_rank_conditional_on_exact_layout'],
             'baseline_abstained': True,
             'oracle_id_exact_generated': oracle_generated,
             'oracle_id_exact_retained': oracle['exact_layout_rank'] is not None,
             'oracle_id_exact_rank': oracle['exact_layout_rank'],
             'oracle_id_affine_raw_agreement': oracle['held_out_affine_raw_agreement']}
    selections = {
        'baseline': [family_key(g) for g in base_groups],
        'expanded_search': [family_key(g) for g in unary_groups],
        'relation_fixed_pool': [family_key(g) for g in fixed_groups],
        'relation_with_reopening': [family_key(g) for g in feedback_groups]}
    return {'schema': SCHEMA, 'vehicle': base['vehicle'], 'hidden_role': role,
            'assessable': bool(references), 'adaptive_budget': budget,
            'candidate_families_scored': len(families),
            'relation_gate': {'status': 'eligible' if eligible else 'unavailable',
                              'source_gate': gate['status'],
                              'source_supporting_vehicles': gate['supporting_vehicles'],
                              'target_anchor': asdict(anchor) if anchor else None},
            'detailed_family_ids': selections,
            'variants': {'baseline': baseline, 'expanded_search': expanded,
                         'relation_fixed_pool': fixed,
                         'relation_with_reopening': feedback},
            'transition_trace': trace, 'oracle_id': oracle,
            'relation_family_scores': [
                {'family': family_key(row),
                 'score': relation_scores[family_key(row)]}
                for row in families] if eligible else [],
            'reference_layouts_evaluation_only': [asdict(f) for f in references]}
