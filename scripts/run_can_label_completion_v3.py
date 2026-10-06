"""Controlled development experiment for ID allocation and relational feedback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from can_discovery.decoding.signals import discover_fields
from can_discovery.label_completion.controlled_v3 import (
    CUTOFF_GAP, EXPANDED_DETAIL, RELATION_WEIGHT, SCHEMA, run_case, source_gate)
from can_discovery.label_completion.manifest import (
    artifact_hash, file_hash, verify_manifest, write_artifact)
from can_discovery.label_completion.pipeline import ROLES
from can_discovery.preprocessing.capture import load_capture


def run(manifest_path: Path, baseline_dir: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest)  # No final-route read or hash check.
    baseline_path = baseline_dir / 'summary-v2.1.json'
    baseline = json.loads(baseline_path.read_text())
    if baseline['manifest_hash'] != artifact_hash(manifest):
        raise ValueError('v2.1 baseline manifest mismatch')
    case_index = {(row['vehicle'], row['role']): Path(row['path'])
                  for row in baseline['cases']}
    if len(case_index) != len(manifest['vehicles'])*len(ROLES):
        raise ValueError('incomplete v2.1 baseline')
    loaded = {}
    for entry in manifest['vehicles']:
        loaded[entry['vehicle']] = (
            [load_capture(Path(item['path'])) for item in entry['routes']['fitting']],
            [load_capture(Path(item['path'])) for item in entry['routes']['development']],
            Path(entry['dbc']['path']))
    gates = {vehicle: source_gate(vehicle, loaded) for vehicle in loaded}
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for entry in manifest['vehicles']:
        vehicle = entry['vehicle']
        fitting, development, dbc = loaded[vehicle]
        registry = discover_fields(fitting, max_ids=64, max_layouts=8)
        for role in ROLES:
            base_path = case_index[(vehicle, role)]
            base = json.loads(base_path.read_text())
            if base['manifest_hash'] != artifact_hash(manifest):
                raise ValueError('case manifest mismatch')
            case = run_case(base, fitting, development, dbc, registry, gates[vehicle])
            case['manifest_hash'] = artifact_hash(manifest)
            case['v2_1_case_sha256'] = file_hash(base_path)
            case['dbc_sha256'] = entry['dbc']['sha256']
            case['fitting_hashes'] = [c.sha256 for c in fitting]
            case['development_hashes'] = [c.sha256 for c in development]
            path = output_dir / f'{vehicle.lower()}-{role}-controlled-v3.json'
            write_artifact(path, case)
            cases.append({'vehicle': vehicle, 'role': role, 'assessable': case['assessable'],
                          'path': str(path), 'adaptive_budget': case['adaptive_budget'],
                          'gate_status': case['relation_gate']['status'],
                          'variants': case['variants'],
                          'transition_trace': case['transition_trace']})
    assessable = [case for case in cases if case['assessable']]
    variants = {}
    for name in ('baseline', 'expanded_search', 'relation_fixed_pool',
                 'relation_with_reopening'):
        variants[name] = {'assessable': len(assessable),
            'id_top_1': sum(c['variants'][name]['id_top_1'] for c in assessable),
            'id_top_5': sum(c['variants'][name]['id_top_5'] for c in assessable),
            'id_top_8': sum(c['variants'][name]['id_top_8'] for c in assessable),
            'exact_layout_top_1': sum(c['variants'][name]['exact_layout_top_1']
                                      for c in assessable),
            'exact_layout_top_8': sum(c['variants'][name]['exact_layout_top_8']
                                      for c in assessable),
            'affine_raw_agreement': sum(c['variants'][name]['held_out_affine_raw_agreement']
                                        for c in assessable)}
    summary = {'schema': SCHEMA, 'kind': 'development_summary',
        'manifest_hash': artifact_hash(manifest),
        'baseline_summary_sha256': file_hash(baseline_path),
        'source_gates': gates, 'cases': cases, 'variants': variants,
        'rules': {'base_detailed_ids': 8, 'expanded_detailed_ids': EXPANDED_DETAIL,
                  'cutoff_gap': CUTOFF_GAP, 'relation_weight': RELATION_WEIGHT,
                  'source_only_gate': True, 'hidden_reference_for_oracle_only': True},
        'final_routes_opened': False, 'final_assessment_allowed': False}
    write_artifact(output_dir / 'summary-v3.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('baseline_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.baseline_dir, args.output_dir)['variants'],
                     sort_keys=True))


if __name__ == '__main__':
    main()
