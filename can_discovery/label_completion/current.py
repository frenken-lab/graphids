"""Progressive full-pool label completion, one vehicle-role case per task."""
from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from ..decoding.dbc import load_dbc_fields
from ..decoding.signals import FieldCandidate
from ..preprocessing.capture import load_capture
from .baseline import (
    DBC_NAMES,
    ROLES,
    _active,
    _score,
    _series,
    _value_equivalent,
    key,
    layout_key,
)
from .full_search import case_specs, verify_staged_manifest
from .manifest import (
    artifact_hash,
    file_hash,
    verify_manifest,
    write_artifact,
)
from .progressive_v6 import (
    RELATION_MIN_NET,
    RELATION_SOURCE_VEHICLES,
    RELATION_WEIGHT,
    SCHEMA,
    anchor_series,
    candidate_features,
    decoded_values,
    field_from_row,
    fit_contrast,
    relation_net,
    trajectory_signature,
)


def _family(row: dict) -> tuple:
    return row['bus'], row['arb_id'], row['payload_length']


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


def _cluster_families(families: list[dict], fitting) -> list[dict]:
    """Group value-equivalent top family representatives; preserve all IDs."""
    parent = list(range(len(families)))
    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for i, left in enumerate(families):
        for j in range(i + 1, len(families)):
            right = families[j]
            if _family(left) == _family(right):
                continue
            if _value_equivalent(field_from_row(left['best_layout']),
                                 field_from_row(right['best_layout']), fitting):
                parent[root(j)] = root(i)
    groups = defaultdict(list)
    for i, family in enumerate(families):
        groups[root(i)].append({'bus': family['bus'], 'arb_id': family['arb_id'],
                                'payload_length': family['payload_length'],
                                'family_rank': family['joint_family_rank']})
    return [sorted(members, key=lambda item: item['family_rank'])
            for members in groups.values() if len(members) > 1]


def run(manifest_path: Path, source_manifest_path: Path, full_dir: Path,
        baseline_dir: Path, output_dir: Path, case_index: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    verify_staged_manifest(source_manifest, manifest)
    verify_manifest(manifest)  # Final files are not read.
    entry, role = case_specs(manifest)[case_index]
    target = entry['vehicle']
    loaded = {item['vehicle']: (
        [load_capture(Path(row['path'])) for row in item['routes']['fitting']],
        [load_capture(Path(row['path'])) for row in item['routes']['development']],
        Path(item['dbc']['path'])) for item in manifest['vehicles']}
    fitting, _, _ = loaded[target]
    v5_path = full_dir / f'{target.lower()}-{role}-prepared.json'
    v5 = json.loads(v5_path.read_text())
    if v5['schema'] != 'dbc-label-completion-preparation-v1':
        raise ValueError('preparation schema mismatch')
    if v5['manifest_hash'] != artifact_hash(source_manifest):
        raise ValueError('v5 manifest mismatch')
    if file_hash(Path(v5['candidate_pool_path'])) != v5['candidate_pool_sha256']:
        raise ValueError('candidate pool hash changed')
    contrast, contrast_evidence = contrast_gate(target, role, loaded, full_dir, baseline_dir)
    anchor, relation_evidence = relation_gate(target, role, loaded)
    target_anchor_samples = anchor_series(fitting, anchor) if anchor else None
    target_family_samples = {}
    groups = defaultdict(dict)
    pool_count = 0
    with gzip.open(v5['candidate_pool_path'], 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            family = _family(row)
            if family not in target_family_samples:
                target_family_samples[family] = _series(fitting, family)
            samples = target_family_samples[family]
            series = decoded_values(row, samples)
            signature = trajectory_signature(series)
            features = candidate_features(row, series, row['score'])
            contrast_vote = contrast.adjustment(features) if contrast else 0.
            relation = relation_net(series, target_anchor_samples) if anchor else None
            relation_vote = (RELATION_WEIGHT * relation if relation is not None else 0.)
            joint = row['score'] + contrast_vote + relation_vote
            classes = groups[family]
            bucket = classes.setdefault(signature, {'members': [], 'best': None,
                'best_score': -math.inf, 'best_unary_score': -math.inf,
                'best_contrast_score': -math.inf, 'contrast_vote': 0.,
                'relation_net': None})
            bucket['members'].append((row['layout_rank'], joint))
            bucket['best_unary_score'] = max(bucket['best_unary_score'], row['score'])
            bucket['best_contrast_score'] = max(
                bucket['best_contrast_score'], row['score'] + contrast_vote)
            if joint > bucket['best_score']:
                bucket.update({'best': row, 'best_score': joint,
                               'contrast_vote': contrast_vote,
                               'relation_net': relation})
            pool_count += 1
    if pool_count != v5['scored_layout_count']:
        raise ValueError('v5 candidate pool count mismatch')
    family_rows = []
    class_rows = []
    candidate_ranks = {}
    for old in v5['families']:
        family = _family(old)
        ranked = sorted(groups[family].items(), key=lambda item:
            (-item[1]['best_score'], item[0]))
        offset = 0
        for class_rank, (signature, bucket) in enumerate(ranked, 1):
            members = sorted(bucket['members'], key=lambda item: (-item[1], item[0]))
            class_rows.append({'family': family, 'class_rank': class_rank,
                'signature': signature, 'best_score': bucket['best_score'],
                'best_unary_score': bucket['best_unary_score'],
                'best_contrast_score': bucket['best_contrast_score'],
                'representative': bucket['best'],
                'contrast_vote': bucket['contrast_vote'],
                'relation_net': bucket['relation_net'],
                'member_layout_ranks': [rank for rank, _ in members]})
            for within, (old_rank, _) in enumerate(members, 1):
                candidate_ranks[(family, old_rank)] = (class_rank, within,
                                                       offset + within)
            offset += len(members)
        best = ranked[0][1] if ranked else None
        all_classes = list(groups[family].values())
        family_rows.append({'bus': old['bus'], 'arb_id': old['arb_id'],
            'payload_length': old['payload_length'],
            'initial_rank': old['family_rank'],
            'unary_family_score': max((row['best_unary_score'] for row in all_classes),
                                      default=None),
            'contrast_family_score': max((row['best_contrast_score'] for row in all_classes),
                                         default=None),
            'joint_score': best['best_score'] if best else None,
            'class_count': len(ranked),
            'best_layout': best['best'] if best else None,
            'candidate_count': old['candidate_count']})
    family_rows.sort(key=lambda row: (
        -(row['joint_score'] if row['joint_score'] is not None else -math.inf),
        repr(_family(row))))
    for i, row in enumerate(family_rows, 1):
        row['joint_family_rank'] = i
    for stage, score_key in [('unary', 'unary_family_score'),
                             ('contrast', 'contrast_family_score')]:
        ordered = sorted(family_rows, key=lambda row: (
            -(row[score_key] if row[score_key] is not None else -math.inf),
            repr(_family(row))))
        for i, row in enumerate(ordered, 1):
            row[f'{stage}_family_rank'] = i
    # All candidate scores and ID ranks are fixed before hidden-reference assessment.
    output_dir.mkdir(parents=True, exist_ok=True)
    class_path = output_dir / f'{target.lower()}-{role}-classes.jsonl.gz'
    temporary = class_path.with_name(class_path.name + f'.tmp-{os.getpid()}')
    try:
        with temporary.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as zipped:
                for row in class_rows:
                    zipped.write((json.dumps(row, sort_keys=True,
                        separators=(',', ':'), allow_nan=False) + '\n').encode())
        if class_path.exists():
            if file_hash(class_path) != file_hash(temporary):
                raise ValueError('immutable class artifact changed')
            temporary.unlink()
        else:
            temporary.replace(class_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    baseline = json.loads((baseline_dir / f'{target.lower()}-{role}-baseline.json').read_text())
    if baseline['schema'] != 'dbc-label-completion-baseline-control-v1' or baseline['manifest_hash'] != artifact_hash(source_manifest):
        raise ValueError('baseline provenance mismatch')
    truths = [FieldCandidate(**item) for item in
              baseline['dbc_annotations']['reference_layouts']]
    exact = []
    old_by_family = {_family(row): row for row in v5['families']}
    new_by_family = {_family(row): row for row in family_rows}
    with gzip.open(v5['candidate_pool_path'], 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if any(layout_key(field_from_row(row)) == layout_key(truth)
                   for truth in truths):
                family = _family(row)
                class_rank, within, flat_rank = candidate_ranks[(family, row['layout_rank'])]
                role_scores = {}
                for candidate_role in ROLES:
                    score, _ = _score(field_from_row(row), target_family_samples[family],
                                      candidate_role)
                    role_scores[candidate_role] = score
                role_order = sorted(ROLES, key=lambda name: (-role_scores[name], name))
                exact.append({'bus': row['bus'], 'arb_id': row['arb_id'],
                    'payload_length': row['payload_length'],
                    'old_id_rank': old_by_family[family]['family_rank'],
                    'unary_id_rank': new_by_family[family]['unary_family_rank'],
                    'contrast_id_rank': new_by_family[family]['contrast_family_rank'],
                    'new_id_rank': new_by_family[family]['joint_family_rank'],
                    'old_layout_rank': row['layout_rank'],
                    'new_layout_rank': flat_rank,
                    'class_rank': class_rank, 'within_class_rank': within,
                    'unary_role_rank': role_order.index(role) + 1,
                    'unary_role_scores': role_scores})
    best_exact = min(exact, key=lambda row: (row['new_id_rank'], row['class_rank'],
                                            row['within_class_rank'])) if exact else None
    clusters = _cluster_families([row for row in family_rows if row['best_layout']], fitting)
    result = {'schema': SCHEMA, 'vehicle': target, 'role': role,
        'manifest_hash': artifact_hash(source_manifest),
        'staged_manifest_hash': artifact_hash(manifest),
        'v5_case_sha256': file_hash(v5_path),
        'candidate_pool_sha256': v5['candidate_pool_sha256'],
        'class_artifact_path': str(class_path),
        'class_artifact_sha256': file_hash(class_path),
        'source_contrast': contrast_evidence,
        'relationship_gate': relation_evidence,
        'scored_layout_count': pool_count, 'trajectory_class_count': len(class_rows),
        'families': family_rows,
        'cross_id_value_clusters': clusters,
        'assessable': bool(truths),
        'reference_transitions_evaluation_only': exact,
        'best_reference_evaluation_only': best_exact,
        'final_routes_opened': False,
        'review_ledger': {'status': 'role hypothesis', 'dbc_edit': False,
                          'physical_verification': False, 'abstained': True}}
    write_artifact(output_dir / f'{target.lower()}-{role}-case.json', result)
    return result
