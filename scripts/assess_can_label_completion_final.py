"""One sealed assessment using frozen development ledgers; gate required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from can_discovery.decoding.signals import FieldCandidate
from can_discovery.label_completion.manifest import artifact_hash, file_hash, verify_manifest, write_artifact
from can_discovery.label_completion.pipeline import ROLES, _reference_fields, _value_equivalent, layout_key
from can_discovery.preprocessing.capture import load_capture

METHOD_FILES = ('can_discovery/label_completion/pipeline.py',
                'can_discovery/label_completion/manifest.py',
                'scripts/run_can_label_completion.py')


def freeze(manifest_path: Path, development_dir: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest)
    summary_path = development_dir / 'summary-v2.1.json'
    summary = json.loads(summary_path.read_text())
    if not summary['final_assessment_allowed']:
        raise ValueError('development gate failed; final routes remain sealed')
    if summary['manifest_hash'] != artifact_hash(manifest):
        raise ValueError('manifest changed')
    cases = {str(Path(c['path']).resolve()): file_hash(Path(c['path']))
             for c in summary['cases']}
    result = {'schema': 'dbc-label-completion-freeze-v2.1',
              'manifest_hash': artifact_hash(manifest),
              'development_summary_hash': file_hash(summary_path),
              'case_hashes': cases,
              'method_hashes': {name: file_hash(Path(name)) for name in METHOD_FILES},
              'limits': {'families': 64, 'detailed_families': 8,
                         'layouts_per_family': 8, 'width_bits': [1, 32]},
              'scoring_rules': 'unary_role_score_v1_best_candidate_per_family',
              'final_assessment_count': 1}
    write_artifact(output, result)
    return result


def assess(manifest_path: Path, development_dir: Path, freeze_path: Path,
           output: Path) -> dict:
    frozen = json.loads(freeze_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if frozen['manifest_hash'] != artifact_hash(manifest):
        raise ValueError('manifest changed after freeze')
    for name, digest in frozen['method_hashes'].items():
        if file_hash(Path(name)) != digest:
            raise ValueError(f'method changed after freeze: {name}')
    summary_path = development_dir / 'summary-v2.1.json'
    if file_hash(summary_path) != frozen['development_summary_hash']:
        raise ValueError('development results changed after freeze')
    for name, digest in frozen['case_hashes'].items():
        if file_hash(Path(name)) != digest:
            raise ValueError(f'case changed after freeze: {name}')
    if output.exists():
        raise ValueError('final assessment already exists')
    verify_manifest(manifest, include_final=True)
    results = []
    for entry in manifest['vehicles']:
        fitting = [load_capture(Path(r['path'])) for r in entry['routes']['fitting']]
        final = load_capture(Path(entry['routes']['final'][0]['path']))
        for role in ROLES:
            path = development_dir / f"{entry['vehicle'].lower()}-{role}-case-v2.1.json"
            case = json.loads(path.read_text())
            truth = _reference_fields(Path(entry['dbc']['path']), role, fitting, [final])
            family_set = {(f.bus, f.arb_id.upper(), f.payload_length) for f in truth}
            id_rank = next((r['rank'] for r in case['message_family_hypotheses']
                            if (r['bus'], r['arb_id'], r['payload_length']) in family_set), None)
            exact_rank = None
            raw_agreement = False
            for family in case['detailed_families']:
                if (family['bus'], family['arb_id'], family['payload_length']) not in family_set:
                    continue
                for row in family['candidate_layouts']:
                    candidate = FieldCandidate(**row['layout'])
                    for reference in truth:
                        if layout_key(candidate) == layout_key(reference):
                            exact_rank = min(exact_rank or 999, row['rank'])
                        if not raw_agreement and _value_equivalent(candidate, reference,
                                                                    [*fitting, final]):
                            raw_agreement = True
            results.append({'vehicle': entry['vehicle'], 'role': role,
                'assessable': bool(truth), 'id_rank': id_rank,
                'id_top_5': id_rank is not None and id_rank <= 5,
                'id_top_8': id_rank is not None and id_rank <= 8,
                'exact_layout_rank': exact_rank, 'exact_layout_top_8': exact_rank is not None and exact_rank <= 8,
                'raw_value_agreement': raw_agreement,
                'final_capture_sha256': final.sha256})
    assessed = [r for r in results if r['assessable']]
    result = {'schema': 'dbc-label-completion-final-v2.1',
        'freeze_hash': artifact_hash(frozen), 'cases': results,
        'summary': {'assessable': len(assessed),
                    'id_top_5': sum(r['id_top_5'] for r in assessed),
                    'id_top_8': sum(r['id_top_8'] for r in assessed),
                    'exact_layout_top_8': sum(r['exact_layout_top_8'] for r in assessed),
                    'raw_value_agreement': sum(r['raw_value_agreement'] for r in assessed)}}
    write_artifact(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('development_dir', type=Path)
    parser.add_argument('freeze_file', type=Path)
    parser.add_argument('final_output', type=Path)
    args = parser.parse_args()
    freeze(args.manifest, args.development_dir, args.freeze_file)
    print(json.dumps(assess(args.manifest, args.development_dir,
                            args.freeze_file, args.final_output)['summary']))


if __name__ == '__main__':
    main()
