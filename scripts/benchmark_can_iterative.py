"""Development-only masked-label rounds for the four pinned physical vehicles."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from can_discovery.decoding.dbc import load_dbc_fields
from can_discovery.iterative.rounds import (
    fit_shared_model,
    infer_vehicle,
    promote_round,
    validate_proposals,
)
from can_discovery.manifests.common import artifact_hash, write_artifact
from can_discovery.manifests.iterative import freeze_manifest
from can_discovery.relationships.physical.role_types import ROLES
from can_discovery.relationships.physical.source_calibration import _dbc_role


def _truth(entry):
    fields = load_dbc_fields(Path(entry['dbc']['path']))
    return {role: [f for f in fields if _dbc_role(f.label_hypotheses[0][0]) == role]
            for role in ROLES}


def _exact(candidate, truth):
    return all(candidate[key] == getattr(truth, key) for key in
               ('arb_id', 'payload_length', 'start', 'width', 'endian', 'signed'))


def assess(entry, proposals):
    """The target DBC is opened here, after fitting and frozen proposal writing."""
    truths = _truth(entry)
    result = {}
    for role in ROLES:
        relevant = [proposal['assignments'][role] for proposal in proposals['proposals']]
        rank = next((i for i, item in enumerate(relevant, 1) if item and
                     any(_exact(item['field'], truth) for truth in truths[role])), None)
        result[role] = {'truth_count': len(truths[role]), 'exact_rank': rank,
                        'exact_top1': rank == 1, 'exact_top8': rank is not None,
                        'top1_null': not relevant or relevant[0] is None,
                        'candidate_coverage': rank is not None}
    return result


def run(manifest, output_dir, *, rounds=2, tracks=('source_only', 'target_anchor_assisted')):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact(output_dir / 'manifest.json', manifest)
    report = []
    for entry in manifest['vehicles']:
        vehicle = entry['vehicle']
        shared = fit_shared_model(manifest, vehicle)
        for round_number in range(rounds + 1):
            if shared['round'] != round_number:
                raise AssertionError('round numbering mismatch')
            model_hash = write_artifact(output_dir / f'{vehicle}-round-{round_number}-shared.json', shared)
            validations = []
            before = time.monotonic()
            for track in tracks:
                cases = [('both_hidden', ())] if track == 'source_only' else [
                    ('speed_hidden', ('acceleration',)),
                    ('acceleration_hidden', ('speed',))]
                for case, anchors in cases:
                    proposals = infer_vehicle(entry, shared, track=track, anchor_roles=anchors)
                    stem = f'{vehicle}-round-{round_number}-{track}-{case}'
                    proposal_hash = write_artifact(output_dir / f'{stem}-proposals.json', proposals)
                    validation = validate_proposals(entry, proposals, shared)
                    write_artifact(output_dir / f'{stem}-validation.json', validation)
                    validations.append(validation)
                    report.append({'vehicle': vehicle, 'round': round_number, 'track': track,
                        'case': case, 'model_hash': model_hash, 'proposal_hash': proposal_hash,
                        'roles': assess(entry, proposals), 'candidate_count': proposals['candidate_count'],
                        'runtime_seconds': time.monotonic() - before,
                        'peak_memory_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'factor_availability': list(shared['factor_weights'])})
            if round_number < rounds:
                promoted = promote_round(shared, validations)
                if promoted['factor_weights'] == shared['factor_weights']:
                    break
                shared = promoted
    result = {'schema': 'can-iterative-v1', 'kind': 'development_report',
              'manifest_hash': artifact_hash(manifest), 'final_promotion_claim': False,
              'results': report}
    write_artifact(output_dir / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark_manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 0 or args.rounds > 2:
        parser.error('rounds must be 0, 1, or 2')
    manifest = freeze_manifest(json.loads(args.benchmark_manifest.read_text()))
    result = run(manifest, args.output_dir, rounds=args.rounds)
    print(json.dumps({'report': str(args.output_dir / 'report.json'),
                      'cases': len(result['results']), 'final_promotion_claim': False}))


if __name__ == '__main__':
    main()
