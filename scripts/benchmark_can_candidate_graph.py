"""Run pinned leave-one-vehicle-out physical and candidate-graph folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical.candidate_graph import fit_graph_weights


def _entries(path: Path, forester_dir: Path | None) -> list[dict]:
    entries = json.loads(path.read_text())
    if forester_dir is not None:
        forester = {'vehicle': 'SUBARU_FORESTER',
                'references': [str(forester_dir / 'reference_1.csv'),
                               str(forester_dir / 'reference_2.csv')],
                'held_out': str(forester_dir / 'reference_3.csv'),
                'dbc': str(forester_dir / 'subaru_global_2017.dbc')}
        entries.append(forester)
    if len(entries) < 4 or len({item['vehicle'].casefold() for item in entries}) != len(entries):
        raise ValueError('benchmark requires at least four distinct vehicles')
    for item in entries:
        fitting = item.get('fitting', item.get('references', ()))
        validation = item.get('validation', [item['held_out']] if 'held_out' in item else ())
        final = item.get('final', item.get('held_out'))
        if len(fitting) != 2 or not validation or not final:
            raise ValueError('each vehicle needs two fitting, validation, and final captures')
        if ('fitting' in item or 'validation' in item or 'final' in item) and len(set((*fitting, *validation, final))) != len(fitting) + len(validation) + 1:
            raise ValueError('capture roles must be disjoint')
        for path in (*fitting, *validation, final, item['dbc']):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        if 'routes' in item:
            if [record['capture'] for record in item['routes']] != [*fitting, *validation, final]:
                raise ValueError('manifest capture roles disagree with route records')
            for record in item['routes']:
                path = Path(record['capture'])
                if record.get('vehicle') != item['vehicle'] or record.get('bus') != 0:
                    raise ValueError('manifest vehicle or bus mismatch')
                if not record.get('dataset_revision') or not record.get('opendbc_revision'):
                    raise ValueError('manifest lacks source revisions')
                if len(record.get('compressed_sha256', '')) != 64:
                    raise ValueError('manifest lacks compressed segment hash')
                if hashlib.sha256(path.read_bytes()).hexdigest() != record['converted_sha256']:
                    raise ValueError(f'converted capture hash changed: {path}')
                if len({record['device'] for record in item['routes']}) != len(item['routes']):
                    raise ValueError('manifest routes must use distinct devices')
                if len({record['session'] for record in item['routes']}) != len(item['routes']):
                    raise ValueError('manifest routes must use distinct sessions')
                if hashlib.sha256(Path(item['dbc']).read_bytes()).hexdigest() != record['dbc_sha256']:
                    raise ValueError('DBC hash changed')
    if any('routes' in item for item in entries):
        hashes = [record['converted_sha256'] for item in entries for record in item.get('routes', ())]
        if len(hashes) != len(set(hashes)):
            raise ValueError('source and target captures overlap')
    return entries


def _metrics(result: dict) -> dict:
    evaluation = result['evaluation']
    roles = evaluation['roles']
    return {'exact_layout': evaluation['selected_layout'],
            'cross_byte_order': evaluation['cross_byte_order'],
            'candidate_recall': evaluation['candidate_pool']['active_recall'],
            'named_role_retrieval': {name: {'candidate_exact_rank': item['first_exact_rank'],
                                            'candidate_value_equivalent_rank': item['first_value_equivalent_rank'],
                                            'combination_exact_rank': item['first_exact_combination_rank'],
                                            'combination_value_equivalent_rank': item['first_value_equivalent_combination_rank']}
                                     for name, item in roles.items()},
            'held_out_value_equivalence': {name: item['best_held_out_value_error']
                                           for name, item in roles.items()},
            'abstentions': evaluation['abstentions'],
            'null_roles': evaluation.get('null_roles', []),
            'truth_diagnostics': evaluation.get('truth_diagnostics', []),
            'factor_availability': evaluation.get('factor_availability', []),
            'source_factor_support': evaluation.get('source_factor_support', []),
            'family_weights': evaluation.get('graph_family_weights'),
            'relationship_replication': result.get('held_out_relationships', []),
            'fit_seconds': result['seconds'],
            'peak_memory_kb': result.get('peak_memory_kb')}


def promotion_gate(rows: list[dict], expected_vehicles: int | None = None) -> dict:
    """Apply the predeclared aggregate-gain and per-vehicle regression gate."""
    by_key = {(row['vehicle'], row['variant'], row['layouts']): row for row in rows}
    vehicles = sorted({row['vehicle'] for row in rows})
    expected_vehicles = expected_vehicles or len(vehicles)
    report = {}
    for layouts in (1, 8):
        comparisons = []
        for vehicle in vehicles:
            baseline = by_key.get((vehicle, 'decoding_only', 1))
            candidate = by_key.get((vehicle, 'candidate_graph', layouts))
            if baseline is None or candidate is None:
                continue
            def retrieved(row):
                return sum(any(rank is not None and rank <= 8 for rank in
                    (item['combination_exact_rank'],
                     item['combination_value_equivalent_rank']))
                    for item in row['named_role_retrieval'].values())
            base_order = baseline['cross_byte_order']
            graph_order = candidate['cross_byte_order']
            order_ok = (base_order['assessed'] == 0 or
                (graph_order['assessed'] > 0 and
                 graph_order['accuracy'] >= base_order['accuracy'] - .02))
            comparisons.append({'vehicle': vehicle,
                'layout_recall_gain': candidate['exact_layout']['active_recall'] -
                                      baseline['exact_layout']['active_recall'],
                'named_role_gain': retrieved(candidate) - retrieved(baseline),
                'precision_ok': candidate['exact_layout']['exact_precision'] >=
                                baseline['exact_layout']['exact_precision'] - .02,
                'cross_byte_order_ok': order_ok})
        report[str(layouts)] = {'assessed_vehicles': len(comparisons),
            'comparisons': comparisons,
            'passes': len(comparisons) == expected_vehicles and
                sum(item['layout_recall_gain'] for item in comparisons) > 0 and
                sum(item['named_role_gain'] for item in comparisons) > 0 and
                all(item['precision_ok'] and item['cross_byte_order_ok']
                    for item in comparisons)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark_manifest', type=Path)
    parser.add_argument('--forester-dir', type=Path,
                        default=Path('/tmp/graphids_comma_benchmark'))
    parser.add_argument('--prospective', action='store_true',
                        help='manifest already includes every vehicle and explicit capture roles')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-ids', type=int, default=64)
    parser.add_argument('--only-vehicle', help='run one fold from the manifest')
    args = parser.parse_args()
    entries = _entries(args.benchmark_manifest, None if args.prospective else args.forester_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = args.output_dir / 'source-manifest.json'
    source_manifest.write_text(json.dumps([{'vehicle': item['vehicle'],
        'captures': item.get('fitting', item.get('references')), 'dbc': item['dbc']} for item in entries], indent=2) + '\n')
    full_manifest = args.output_dir / 'benchmark-manifest.json'
    full_manifest.write_text(json.dumps(entries, indent=2) + '\n')
    rows = []
    for target in entries:
        vehicle = target['vehicle']
        if args.only_vehicle and vehicle.casefold() != args.only_vehicle.casefold():
            continue
        sources = [(item['vehicle'],
                    [load_capture(Path(path)) for path in item.get('fitting', item.get('references'))],
                    [load_capture(Path(path)) for path in item.get('validation', [item['held_out']] if 'held_out' in item else ())],
                    Path(item['dbc'])) for item in entries if item is not target]
        started = time.perf_counter()
        artifact = fit_graph_weights(sources, target_vehicle=vehicle)
        artifact_path = args.output_dir / f'{vehicle.lower()}-candidate-graph-v1.json'
        artifact.write(artifact_path)
        artifact_seconds = time.perf_counter() - started
        for variant, layouts in (('decoding_only', 1), ('physical', 1),
                                 ('candidate_graph', 1), ('candidate_graph', 8)):
            output = args.output_dir / f'{vehicle.lower()}-{variant}-layouts{layouts}.json'
            command = [sys.executable, str(Path(__file__).with_name('experiment_can_physical.py')),
                       *target.get('fitting', target.get('references')),
                       target.get('final', target.get('held_out')),
                       '--target-vehicle', vehicle, '--evaluation-dbc', target['dbc'],
                       '--audit-candidate-pool', '--max-ids', str(args.max_ids),
                       '--output', str(output)]
            if variant != 'decoding_only':
                command += ['--source-manifest', str(source_manifest),
                            '--source-benchmark-manifest', str(full_manifest)]
            if variant == 'candidate_graph':
                command += ['--graph-artifact', str(artifact_path), '--max-layouts', str(layouts)]
            subprocess.run(command, check=True)
            result = json.loads(output.read_text())
            row = {'vehicle': vehicle, 'variant': variant, 'layouts': layouts,
                   'artifact_fit_seconds': artifact_seconds,
                   'artifact': str(artifact_path), **_metrics(result)}
            rows.append(row)
            print(json.dumps(row, separators=(',', ':')), flush=True)
            (args.output_dir / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
            (args.output_dir / 'promotion-gate.json').write_text(
                json.dumps(promotion_gate(rows, len(entries)), indent=2) + '\n')


if __name__ == '__main__':
    main()
