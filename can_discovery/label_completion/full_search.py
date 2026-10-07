"""Exhaustive development search, one pinned vehicle-role case per process."""
from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path

from ..decoding.dbc import load_dbc_fields
from ..decoding.signals import (
    _bit_positions,
    discover_fields,
)
from ..preprocessing.capture import load_capture
from .baseline import (
    DBC_NAMES,
    ROLES,
    _protocol,
    _reference_fields,
    _score,
    _series,
    attach_role_ranks,
    enumerate_layouts,
    interpretation_key,
    key,
    layout_key,
    mine_case,
    overlaps,
    rank_families,
)
from .manifest import (
    artifact_hash,
    file_hash,
    verify_manifest,
    write_artifact,
)

SCHEMA = 'dbc-label-completion-preparation-v1'


def score_family(group: dict, role: str, fitting, registry, anchors,
                 max_width: int = 64) -> dict:
    """Score every admissible interpretation without the retired diversity policy."""
    family = group['bus'], group['arb_id'], group['payload_length']
    proposals = [f for f in registry.candidates if key(f) == family
                 and f.role == 'unknown' and f.signed is not None]
    if not proposals:
        raise ValueError(f'no proposals for detailed family {family}')
    protocol = set(group['protocol_bits'])
    samples = _series(fitting, family)
    options = {}
    for field in [*proposals, *enumerate_layouts(proposals[0], protocol,
                                                 max_width=max_width)]:
        if any(overlaps(field, anchor) for anchor in anchors):
            continue
        if _bit_positions(field.start, field.width, field.endian) & protocol:
            continue
        identity = interpretation_key(field)
        if identity in options:
            continue
        score, _ = _score(field, samples, role)
        if math.isfinite(score):
            options[identity] = {'field': field, 'score': float(score)}
    ordered = sorted(options.values(), key=lambda r: (-r['score'],
                     repr(layout_key(r['field']))))
    return {'pool_size': len(ordered), 'ordered': ordered}


def case_specs(manifest: dict) -> list[tuple[dict, str]]:
    return [(entry, role) for entry in manifest['vehicles'] for role in ROLES]


def rank_all_families(registry, fitting, anchors, role: str) -> list[dict]:
    grouped = {}
    for field in registry.candidates:
        if field.role == 'unknown' and field.signed is not None:
            grouped.setdefault(key(field), []).append(field)
    ranked = []
    for family, fields in grouped.items():
        samples = _series(fitting, family)
        for field in fields:
            if any(overlaps(field, anchor) for anchor in anchors):
                continue
            score, _ = _score(field, samples, role)
            if math.isfinite(score):
                ranked.append((field, score))
    return rank_families(ranked)


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
                                  'signed': field.signed, 'score': item['score']}
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


def verify_staged_manifest(source: dict, staged: dict) -> None:
    """Require identical cohort, split order, and pinned content hashes."""
    if len(source['vehicles']) != len(staged['vehicles']):
        raise ValueError('staged manifest vehicle count mismatch')
    for original, copied in zip(source['vehicles'], staged['vehicles']):
        if original['vehicle'] != copied['vehicle']:
            raise ValueError('staged vehicle order mismatch')
        if original['dbc']['sha256'] != copied['dbc']['sha256']:
            raise ValueError('staged DBC hash mismatch')
        for split in ('fitting', 'development', 'final'):
            if [x['sha256'] for x in original['routes'][split]] != [
                    x['sha256'] for x in copied['routes'][split]]:
                raise ValueError(f'staged {split} hash mismatch')
        if original['routes']['final'] != copied['routes']['final']:
            raise ValueError('sealed final route changed in staged manifest')


def run(manifest_path: Path, output_dir: Path, case_index: int,
        source_manifest_path: Path) -> dict:
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
    # A large bound removes the earlier 64-family truncation. The eligible set
    # still follows prepare_traces' common-route and transport filters.
    registry = discover_fields(fitting, max_ids=1_000_000, max_layouts=8)
    annotations = load_dbc_fields(Path(entry['dbc']['path']))
    anchors = [field for field in annotations
               if field.label_hypotheses[0][0].upper() not in DBC_NAMES[role]
               and any(field.label_hypotheses[0][0].upper() in names
                       for names in DBC_NAMES.values())]
    families = rank_all_families(registry, fitting, anchors, role)
    scored_groups = []
    for family in families:
        coordinates = (family['bus'], family['arb_id'], family['payload_length'])
        result = score_family({**family,
            'protocol_bits': sorted(_protocol(registry, coordinates))},
            role, fitting, registry, anchors, max_width=64)
        scored_groups.append((family, result))

    # Candidate ranks are complete before the hidden reference is opened.
    pool_path = output_dir / f'{vehicle.lower()}-{role}-pool.jsonl.gz'
    pool_hash = write_full_pool(pool_path, scored_groups)
    references = list(_reference_fields(Path(entry['dbc']['path']), role, fitting, development))
    baseline_registry = discover_fields(fitting, max_ids=64, max_layouts=8)
    baseline = attach_role_ranks(mine_case(vehicle, role, fitting, development,
                                          Path(entry['dbc']['path']), baseline_registry), fitting)
    baseline['manifest_hash'] = artifact_hash(source_manifest)
    baseline['schema'] = 'dbc-label-completion-baseline-control-v1'
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
    summary = {'schema': SCHEMA, 'case_index': case_index,
        'vehicle': vehicle, 'hidden_role': role,
        'manifest_hash': artifact_hash(source_manifest),
        'staged_manifest_hash': artifact_hash(manifest),
        'dbc_sha256': entry['dbc']['sha256'],
        'fitting_hashes': [c.sha256 for c in fitting],
        'development_hashes': [c.sha256 for c in development],
        'candidate_pool_path': str(pool_path),
        'candidate_pool_sha256': pool_hash,
        'eligible_family_count': len(families),
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
