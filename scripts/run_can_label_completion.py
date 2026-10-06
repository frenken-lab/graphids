"""Build development review ledgers for the pinned five-route DBC cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from can_discovery.decoding.signals import discover_fields
from can_discovery.label_completion.manifest import artifact_hash, verify_manifest, write_artifact
from can_discovery.label_completion.pipeline import ROLES, attach_role_ranks, mine_case
from can_discovery.preprocessing.capture import load_capture


def run(manifest_path: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest)  # Deliberately never opens or hashes sealed final routes.
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for entry in manifest['vehicles']:
        fitting = [load_capture(Path(r['path'])) for r in entry['routes']['fitting']]
        development = [load_capture(Path(r['path'])) for r in entry['routes']['development']]
        registry = discover_fields(fitting, max_ids=64, max_layouts=8)
        for role in ROLES:
            case = attach_role_ranks(mine_case(entry['vehicle'], role, fitting, development,
                                     Path(entry['dbc']['path']), registry), fitting)
            case['manifest_hash'] = artifact_hash(manifest)
            case['dbc_sha256'] = entry['dbc']['sha256']
            case['fitting_hashes'] = [c.sha256 for c in fitting]
            case['development_hashes'] = [c.sha256 for c in development]
            path = output_dir / f"{entry['vehicle'].lower()}-{role}-case-v2.1.json"
            write_artifact(path, case)
            cases.append({'vehicle': entry['vehicle'], 'role': role,
                          'path': str(path), 'metrics': case['metrics'],
                          'assessable': case['dbc_annotations']['assessable']})
    assessable = [c for c in cases if c['assessable']]
    summary = {'schema': 'dbc-label-completion-development-v2.1',
        'manifest_hash': artifact_hash(manifest), 'cases': cases,
        'baseline_v1': {'assessable': 9, 'id_top_1': 1, 'id_top_5': 7,
                        'exact_layout_top_8': 0},
        'metrics': {'assessable': len(assessable),
            'id_top_1': sum(c['metrics']['id_top_1'] for c in assessable),
            'id_top_5': sum(c['metrics']['id_top_5'] for c in assessable),
            'id_top_8': sum(c['metrics']['id_top_8'] for c in assessable),
            'exact_layout_top_1': sum(c['metrics']['exact_layout_top_1'] for c in assessable),
            'exact_layout_top_8': sum(c['metrics']['exact_layout_top_8'] for c in assessable),
            'role_top_1': sum(c['metrics']['role_rank'] == 1 for c in assessable),
            'pool_coverage': sum(c['metrics']['exact_layout_pool_coverage'] for c in assessable),
            'raw_value_agreement': sum(c['metrics']['held_out_raw_value_agreement'] for c in assessable)},
        'final_assessment_allowed': False}
    summary['final_assessment_allowed'] = (summary['metrics']['id_top_5'] >= 7 and
        summary['metrics']['exact_layout_top_8'] >= 1)
    write_artifact(output_dir / 'summary-v2.1.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output_dir)['metrics'], sort_keys=True))


if __name__ == '__main__':
    main()
