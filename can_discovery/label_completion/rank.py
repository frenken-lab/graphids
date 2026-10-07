"""Rank one prepared vehicle-role case within its evidence track."""
from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from ..decoding.signals import FieldCandidate
from ..preprocessing.capture import load_capture
from .baseline import ROLES, _value_equivalent, layout_key
from ..scoring.role_guided import _score, _series
from ..scoring.dbc_assisted import (
    candidate_votes,
    contrast_gate,
    relation_gate,
    target_relation_samples,
)
from ..scoring.composition import rank_evidence, score_record
from ..scoring.physical import physical_factor_evidence
from ..scoring import dbc_assisted
from .case import SCHEMA as PREPARATION_SCHEMA, TRACKS, case_specs
from ..manifests.label_completion import (
    artifact_hash,
    file_hash,
    verify_manifest,
    verify_staged_manifest,
    write_artifact,
)
from .progressive_v6 import (
    SCHEMA,
    decoded_values,
    field_from_row,
    trajectory_signature,
)


def _family(row: dict) -> tuple:
    return row['bus'], row['arb_id'], row['payload_length']


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
        baseline_dir: Path, output_dir: Path, case_index: int,
        track: str = 'dbc_assisted') -> dict:
    if track not in TRACKS:
        raise ValueError(f'unknown evidence track: {track}')
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
    if v5['schema'] != PREPARATION_SCHEMA:
        raise ValueError('preparation schema mismatch')
    if v5['track'] != track:
        raise ValueError('preparation evidence track mismatch')
    if v5['manifest_hash'] != artifact_hash(source_manifest):
        raise ValueError('v5 manifest mismatch')
    if file_hash(Path(v5['candidate_pool_path'])) != v5['candidate_pool_sha256']:
        raise ValueError('candidate pool hash changed')
    if track == 'dbc_assisted':
        contrast, contrast_evidence = contrast_gate(target, role, loaded, full_dir, baseline_dir)
        anchor, relation_evidence = relation_gate(target, role, loaded)
    else:
        contrast, contrast_evidence = None, {'status': 'not_applicable', 'reason': 'unsupervised_track'}
        anchor, relation_evidence = None, {'status': 'not_applicable', 'reason': 'unsupervised_track'}
    target_anchor_samples = target_relation_samples(fitting, anchor)
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
            contrast_vote, relation_vote, relation, contrast_valid = (
                candidate_votes(row, series, contrast, target_anchor_samples)
                if track == 'dbc_assisted' else (0., 0., None, False))
            scoring = score_record(rank_evidence(
                row['score'], contrast_vote, relation_vote, relation,
                assisted=track == 'dbc_assisted', contrast_available=contrast is not None,
                contrast_valid=contrast_valid))
            joint = scoring['score']
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
                bucket.update({'best': {**row, 'joint_score_evidence': scoring['evidence']},
                               'best_score': joint,
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
            'timing_evidence': old['timing_evidence'],
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
    wheel_evidence = None
    if role == 'vehicle_speed' and track == 'dbc_assisted':
        candidate = next((field_from_row(row['best_layout']) for row in family_rows
                          if row['best_layout']), None)
        wheels = [field for field in dbc_assisted.load_dbc_fields(loaded[target][2])
                  if field.label_hypotheses[0][0].upper().startswith('WHEEL_SPEED')]
        if candidate is not None and wheels:
            wheel_evidence = [asdict(physical_factor_evidence(
                fitting, 'wheel_speed', (candidate, wheel))) for wheel in wheels]
        else:
            wheel_evidence = [asdict(physical_factor_evidence(
                fitting, 'wheel_speed', None))]
    result = {'schema': SCHEMA, 'track': track, 'vehicle': target, 'role': role,
        'manifest_hash': artifact_hash(source_manifest),
        'staged_manifest_hash': artifact_hash(manifest),
        'v5_case_sha256': file_hash(v5_path),
        'candidate_pool_sha256': v5['candidate_pool_sha256'],
        'class_artifact_path': str(class_path),
        'class_artifact_sha256': file_hash(class_path),
        'source_contrast': contrast_evidence,
        'relationship_gate': relation_evidence,
        'wheel_speed_evidence_diagnostic_only': wheel_evidence,
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
