"""Run one pinned vehicle-role case for a declared evidence track."""
from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from ..decoding.signals import discover_fields
from ..prepare import prepare_families
from ..scoring.dbc_assisted import filter_known_role_anchors, other_role_anchors
from ..scoring.id_priority import motion_id_priority
from ..scoring.role_guided import score_family
from ..scoring.timing import arrival_timing_evidence
from ..preprocessing.capture import load_capture
from .baseline import (
    ROLES,
    _reference_fields,
    attach_role_ranks,
    key,
    layout_key,
    mine_case,
)
from ..manifests.label_completion import (
    artifact_hash,
    file_hash,
    verify_manifest,
    verify_staged_manifest,
    write_artifact,
)

SCHEMA = 'dbc-label-completion-preparation-v4'
TRACKS = ('unsupervised', 'dbc_assisted')


def case_specs(manifest: dict) -> list[tuple[dict, str]]:
    return [(entry, role) for entry in manifest['vehicles'] for role in ROLES]


def write_full_pool(path: Path, groups: list[tuple[dict, dict]]) -> str:
    """Write every scored interpretation in stable rank order, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temporary.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0,
                               compresslevel=6) as stream:
                for family_row, result in groups:
                    for rank, item in enumerate(result['ordered'], 1):
                        field = item['field']
                        record = {'family_rank': family_row['rank'],
                                  'bus': field.bus, 'arb_id': field.arb_id,
                                  'payload_length': field.payload_length,
                                  'layout_rank': rank, 'start': field.start,
                                  'width': field.width, 'endian': field.endian,
                                  'signed': field.signed, 'score': item['score'],
                                  'score_evidence': item['evidence']}
                        stream.write((json.dumps(record, sort_keys=True,
                            separators=(',', ':'), allow_nan=False) + '\n').encode())
        digest = file_hash(temporary)
        if path.exists():
            if file_hash(path) != digest:
                raise ValueError(f'immutable candidate pool already exists: {path}')
            temporary.unlink()
        else:
            temporary.replace(path)
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()


def run(manifest_path: Path, output_dir: Path, case_index: int,
        source_manifest_path: Path, track: str = 'dbc_assisted') -> dict:
    if track not in TRACKS:
        raise ValueError(f'unknown evidence track: {track}')
    manifest = json.loads(manifest_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    verify_staged_manifest(source_manifest, manifest)
    verify_manifest(manifest)  # Final routes are neither read nor hash-checked.
    specs = case_specs(manifest)
    if not 0 <= case_index < len(specs):
        raise ValueError(f'case index must be 0..{len(specs)-1}')
    entry, role = specs[case_index]
    vehicle = entry['vehicle']
    fitting = [load_capture(Path(item['path'])) for item in entry['routes']['fitting']]
    development = [load_capture(Path(item['path'])) for item in entry['routes']['development']]
    # Eligibility is capture-based; a tokenizer cannot admit or reject a family.
    prepared_families = prepare_families(fitting)
    id_priority = (motion_id_priority(prepared_families, fitting)
                   if role == 'vehicle_speed' else [])
    priority_by_family = {(row['bus'], row['arb_id'], row['payload_length']): row
                          for row in id_priority}
    if id_priority:
        prepared_families = tuple(sorted(prepared_families, key=lambda item:
            priority_by_family[(item.trace.bus, item.trace.arb_id.upper(),
                                item.trace.payload_length)]['rank']))
    anchors = (other_role_anchors(Path(entry['dbc']['path']), role)
               if track == 'dbc_assisted' else ())
    scored_groups = []
    for prepared in prepared_families:
        trace = prepared.trace
        family = {'bus': trace.bus, 'arb_id': trace.arb_id,
                  'payload_length': trace.payload_length,
                  'protocol_bits': sorted(prepared.protocol_bits)}
        layouts = prepared.layouts(max_width=64)
        if track == 'dbc_assisted':
            layouts = filter_known_role_anchors(layouts, family, anchors)
        result = score_family(family, layouts, role, fitting)
        if result['ordered']:
            scored_groups.append((family, result))
    scored_groups.sort(key=lambda item: (-item[1]['ordered'][0]['score'],
                        repr((item[0]['bus'], item[0]['arb_id'],
                              item[0]['payload_length']))))
    families = []
    for rank, (family, result) in enumerate(scored_groups, 1):
        family['rank'] = rank
        family['score'] = result['ordered'][0]['score']
        families.append(family)

    # Candidate ranks are complete before the hidden reference is opened.
    pool_path = output_dir / f'{vehicle.lower()}-{role}-pool.jsonl.gz'
    pool_hash = write_full_pool(pool_path, scored_groups)
    references = list(_reference_fields(Path(entry['dbc']['path']), role, fitting, development))
    reference_priority_rank = min((row['rank'] for row in id_priority
        if any((row['bus'], row['arb_id'], row['payload_length']) == key(field)
               for field in references)), default=None)
    baseline_registry = discover_fields(fitting, max_ids=64, max_layouts=8)
    baseline = attach_role_ranks(mine_case(vehicle, role, fitting, development,
                                          Path(entry['dbc']['path']), baseline_registry), fitting)
    baseline['manifest_hash'] = artifact_hash(source_manifest)
    baseline['schema'] = 'dbc-label-completion-baseline-control-v1'
    baseline['track'] = 'dbc_assisted'
    write_artifact(output_dir / f'{vehicle.lower()}-{role}-baseline.json', baseline)
    family_rows = []
    for family, result in scored_groups:
        family_key = (family['bus'], family['arb_id'], family['payload_length'])
        truths = [field for field in references if key(field) == family_key]
        exact_ranks = [i for i, row in enumerate(result['ordered'], 1)
                       if any(layout_key(row['field']) == layout_key(t)
                              for t in truths)]
        family_rows.append({'family_rank': family['rank'],
            'bus': family['bus'], 'arb_id': family['arb_id'],
            'payload_length': family['payload_length'],
            'id_priority_rank': priority_by_family.get(
                (family['bus'], family['arb_id'].upper(), family['payload_length']), {}).get('rank'),
            'id_priority_score': priority_by_family.get(
                (family['bus'], family['arb_id'].upper(), family['payload_length']), {}).get('priority'),
            'timing_evidence': asdict(arrival_timing_evidence(fitting, family_key)),
            'retrieval_score': family['score'],
            'candidate_count': result['pool_size'],
            'reference_exact_rank_evaluation_only': min(exact_ranks) if exact_ranks else None,
            'reference_family_evaluation_only': bool(truths),
            'top_layouts': [{'rank': i, 'score': row['score'],
                'start': row['field'].start, 'width': row['field'].width,
                'endian': row['field'].endian, 'signed': row['field'].signed}
                for i, row in enumerate(result['ordered'][:8], 1)]})
    reference_id_rank = min((row['family_rank'] for row in family_rows
                             if row['reference_family_evaluation_only']), default=None)
    reference_layout_rank = min((row['reference_exact_rank_evaluation_only']
                                  for row in family_rows
                                  if row['reference_exact_rank_evaluation_only'] is not None),
                                 default=None)
    summary = {'schema': SCHEMA, 'track': track, 'case_index': case_index,
        'vehicle': vehicle, 'hidden_role': role,
        'manifest_hash': artifact_hash(source_manifest),
        'staged_manifest_hash': artifact_hash(manifest),
        'dbc_sha256': entry['dbc']['sha256'],
        'fitting_hashes': [c.sha256 for c in fitting],
        'development_hashes': [c.sha256 for c in development],
        'candidate_pool_path': str(pool_path),
        'candidate_pool_sha256': pool_hash,
        'eligible_family_count': len(families),
        'id_priority': {'method': 'cross_capture_motion_activity_v1',
                        'role': 'vehicle_speed', 'families': id_priority,
                        'reference_rank_evaluation_only': reference_priority_rank,
                        'reference_in_top_32_evaluation_only': (
                            reference_priority_rank <= 32
                            if reference_priority_rank is not None else None)}
                       if id_priority else None,
        'scored_layout_count': sum(result['pool_size'] for _, result in scored_groups),
        'layout_width_limit': 64,
        'assessable': bool(references),
        'reference_id_rank_evaluation_only': reference_id_rank,
        'reference_exact_layout_rank_evaluation_only': reference_layout_rank,
        'families': family_rows,
        'final_routes_opened': False,
        'status': 'exhaustive_candidate_ranking; physical_role_not_verified'}
    write_artifact(output_dir / f'{vehicle.lower()}-{role}-prepared.json', summary)
    return summary
