"""Fit an experimental signal graph on declared fitting and validation routes.

Final captures and target DBCs are never opened by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from can_discovery.cohort import load_graph_cohort, split_paths
from can_discovery.decoding.dbc import evaluate_discovery, load_dbc_fields
from can_discovery.decoding.field_values import extract
from can_discovery.decoding.signals import (
    FieldCandidate,
    _bit_positions,
    discover_fields,
)
from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical import (
    ROLES,
    _dbc_role,
    fit_physical_roles,
    fit_source_calibration,
    fit_source_factor_validation,
)
from can_discovery.relationships.physical.signal_graph import (
    build_signal_graph,
)


def run(entries: list[dict], target: dict, *, max_ids: int = 64) -> dict:
    vehicle = target['vehicle']
    fitting_paths = split_paths(target, 'fitting')
    validation_paths = split_paths(target, 'validation')
    fitting = [load_capture(Path(path)) for path in fitting_paths]
    validation = [load_capture(Path(path)) for path in validation_paths]
    if len({c.sha256 for c in (*fitting, *validation)}) != len(fitting)+len(validation):
        raise ValueError('fitting and validation captures must be distinct')
    sources = [(entry['vehicle'],
                [load_capture(Path(path)) for path in split_paths(entry, 'fitting')],
                [load_capture(Path(path)) for path in split_paths(entry, 'validation')],
                Path(entry['dbc'])) for entry in entries if entry is not target]
    target_hashes = {c.sha256 for c in (*fitting, *validation)}
    if any(c.sha256 in target_hashes for _, source_fits, source_validation, _ in sources
           for c in (*source_fits, *source_validation)):
        raise ValueError('source and target capture hashes overlap')
    started = time.perf_counter()
    calibration = fit_source_calibration([(vehicle, fits, dbc)
                                          for vehicle, fits, _, dbc in sources],
                                         target_vehicle=vehicle)
    factors = fit_source_factor_validation(sources, target_vehicle=vehicle)
    registry = discover_fields(fitting, max_ids=max_ids, max_layouts=8)
    discovery = fit_physical_roles(fitting, max_ids=max_ids,
                                   candidates=registry.candidates,
                                   source_calibration=calibration,
                                   validated_factors=factors)
    result = build_signal_graph([*fitting, *validation], registry, discovery)
    result.update({'vehicle': vehicle, 'fitting_hashes': [c.sha256 for c in fitting],
                   'validation_hashes': [c.sha256 for c in validation],
                   'selected_fields': [asdict(field) for field in registry.selected],
                   'candidate_fields': [asdict(field) for field in registry.candidates],
                   'rejected_candidates': list(discovery.rejected),
                   'factor_evidence': [asdict(item) for item in discovery.factor_evidence],
                   'source_calibration': [asdict(item) for item in calibration],
                   'source_factor_validation': [asdict(item) for item in factors],
                   'source_vehicles': [item[0] for item in sources],
                   'source_provenance': [{'vehicle': source_vehicle,
                       'fitting_hashes': [c.sha256 for c in source_fits],
                       'validation_hashes': [c.sha256 for c in source_validation],
                       'dbc_sha256': hashlib.sha256(dbc.read_bytes()).hexdigest()}
                       for source_vehicle, source_fits, source_validation, dbc in sources],
                   'fit_seconds': time.perf_counter()-started,
                   'promotion_status': 'experimental_unpromoted'})
    return result


def evaluate(result: dict, target: dict) -> dict:
    """Open the target DBC only after fitting has returned."""
    expected = {route['dbc_sha256'] for route in target.get('routes', ())
                if 'dbc_sha256' in route}
    if expected and (len(expected) != 1 or
                     hashlib.sha256(Path(target['dbc']).read_bytes()).hexdigest()
                     not in expected):
        raise ValueError('target DBC hash differs from frozen manifest')
    truth = load_dbc_fields(Path(target['dbc']))
    validation_paths = split_paths(target, 'validation')
    validation = load_capture(Path(validation_paths[0]))
    fitting = [load_capture(Path(path)) for path in split_paths(target, 'fitting')]
    def payload_index(captures):
        grouped = {}
        for capture in captures:
            for frame in capture.frames:
                if frame.payload is not None:
                    grouped.setdefault((frame.arb_id.upper(), len(frame.payload)),
                                       []).append(frame.payload)
        return {key: values[::max(1, len(values)//512)][:512]
                for key, values in grouped.items()}
    training_payloads = payload_index(fitting)
    validation_payloads = payload_index([validation])
    fields = tuple(FieldCandidate(**record) for record in result['fields'])
    selected = tuple(FieldCandidate(**record) for record in result['selected_fields'])
    pool = tuple(FieldCandidate(**record) for record in result['candidate_fields'])
    layout = evaluate_discovery(selected, truth, validation)
    candidate = evaluate_discovery(pool, truth, validation)
    def exact_truth_role(field):
        for reference in truth:
            if (field.arb_id.upper() == reference.arb_id.upper() and
                field.payload_length == reference.payload_length and
                field.mux_start == reference.mux_start and
                field.mux_value == reference.mux_value and
                field.endian == reference.endian and
                field.signed == reference.signed and
                _bit_positions(field.start, field.width, field.endian) ==
                _bit_positions(reference.start, reference.width, reference.endian)):
                return _dbc_role(reference.label_hypotheses[0][0])
        return None
    truth_roles = [exact_truth_role(field) for field in fields]
    edge_assessment = {'assessed': 0, 'same_role': 0, 'different_role': 0,
                       'unassessable': 0}
    for edge in result['field_edges']:
        if edge['status'] != 'supported':
            continue
        left, right = truth_roles[edge['source']], truth_roles[edge['target']]
        if left is None or right is None:
            edge_assessment['unassessable'] += 1
        else:
            edge_assessment['assessed'] += 1
            edge_assessment['same_role' if left == right else 'different_role'] += 1
    named = {}
    def equivalent(field, reference):
        key = (reference.arb_id.upper(), reference.payload_length)
        if (field.arb_id.upper(), field.payload_length) != key:
            return False
        if (field.mux_start, field.mux_value) != (reference.mux_start,
                                                  reference.mux_value):
            return False
        def paired(index):
            pairs = []
            for payload in index.get(key, ()):
                if field.mux_start is not None and \
                        payload[field.mux_start//8] != field.mux_value:
                    continue
                if reference.mux_start is not None and \
                        payload[reference.mux_start//8] != reference.mux_value:
                    continue
                pairs.append((extract(payload, field.start, field.width,
                                      field.endian, field.signed),
                              extract(payload, reference.start, reference.width,
                                      reference.endian, reference.signed)))
            return np.asarray(pairs, dtype=float).reshape(-1, 2)
        train, test = paired(training_payloads), paired(validation_payloads)
        if len(train) < 30 or len(test) < 30 or min(np.ptp(train, axis=0)) == 0:
            return False
        slope, offset = np.polyfit(train[:,0], train[:,1], 1)
        span = float(np.ptp(test[:,1]))
        return bool(slope > 0 and span > 0 and np.quantile(np.abs(
            test[:,1]-(slope*test[:,0]+offset)), .99) <= max(1., .01*span))
    for role in ROLES:
        references = [field for field in truth if
                      _dbc_role(field.label_hypotheses[0][0]) == role]
        groups = [group for group in result['role_groups'] if group['role'] == role]
        exact_rank = None
        value_rank = None
        root_rank = None
        for rank, group in enumerate(groups, 1):
            def exact(index, references=references):
                field = fields[index]
                return any(field.arb_id.upper() == ref.arb_id.upper() and
                    field.payload_length == ref.payload_length and
                    field.mux_start == ref.mux_start and
                    field.mux_value == ref.mux_value and
                    field.endian == ref.endian and field.signed == ref.signed and
                    _bit_positions(field.start, field.width, field.endian) ==
                    _bit_positions(ref.start, ref.width, ref.endian)
                    for ref in references)
            if exact_rank is None and any(map(exact, group['members'])):
                exact_rank = rank
            if value_rank is None and any(equivalent(fields[index], ref)
                for index in group['members'] for ref in references):
                value_rank = rank
            if root_rank is None and any(map(exact, group['observed_roots'])):
                root_rank = rank
        named[role] = {'dbc_fields': len(references), 'first_exact_group_rank': exact_rank,
                       'first_value_equivalent_group_rank': value_rank,
                       'first_exact_root_rank': root_rank,
                       'retrieved_top_8': any(rank is not None and rank <= 8
                           for rank in (exact_rank, value_rank))}
    return {'dbc_sha256': hashlib.sha256(Path(target['dbc']).read_bytes()).hexdigest(),
            'selected_layout': layout, 'candidate_pool': candidate,
            'named_roles': named,
            'field_edge_semantics': edge_assessment,
            'supported_field_edges': sum(edge['status'] == 'supported'
                                         for edge in result['field_edges']),
            'supported_role_edges': sum(edge['status'] == 'supported'
                                        for edge in result['role_edges']),
            'null_roles': result['null_roles']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark_manifest', type=Path)
    parser.add_argument('--forester-dir', type=Path,
                        default=Path('/tmp/graphids_comma_benchmark'))
    parser.add_argument('--prospective', action='store_true')
    parser.add_argument('--only-vehicle')
    parser.add_argument('--max-ids', type=int, default=64)
    parser.add_argument('--evaluate-dbc', action='store_true',
                        help='open target DBC only after the graph is fully fitted')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    entries = load_graph_cohort(args.benchmark_manifest,
                       None if args.prospective else args.forester_dir, policy='signal_graph')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for target in entries:
        vehicle = target['vehicle']
        if args.only_vehicle and vehicle.casefold() != args.only_vehicle.casefold():
            continue
        result = run(entries, target, max_ids=args.max_ids)
        if args.evaluate_dbc:
            result['evaluation'] = evaluate(result, target)
        output = args.output_dir / f'{vehicle.lower()}-signal_graph-v2.json'
        output.write_text(json.dumps(result, indent=2) + '\n')
        summary.append({'vehicle': vehicle, 'artifact': str(output),
                        'fields': len(result['fields']),
                        'field_edges': len(result['field_edges']),
                        'supported_field_edges': sum(e['status'] == 'supported'
                            for e in result['field_edges']),
                        'role_groups': len(result['role_groups']),
                        'supported_role_edges': sum(e['status'] == 'supported'
                            for e in result['role_edges']),
                        'null_roles': result['null_roles'],
                        'fit_seconds': result['fit_seconds']})
        if 'evaluation' in result:
            summary[-1]['candidate_pool'] = {key: result['evaluation']['candidate_pool'][key]
                for key in ('candidate_count', 'dbc_signal_count', 'exact_count',
                            'precision', 'recall')}
            summary[-1]['selected_layout'] = {key: result['evaluation']['selected_layout'][key]
                for key in ('candidate_count', 'dbc_signal_count', 'exact_count',
                            'precision', 'recall')}
            summary[-1]['named_roles'] = result['evaluation']['named_roles']
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary[-1]), flush=True)


if __name__ == '__main__':
    main()
