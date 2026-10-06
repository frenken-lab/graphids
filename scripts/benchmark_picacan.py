"""Run the isolated PicaCAN baseline on a frozen cohort manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from can_discovery.decoding.dbc import evaluate_discovery, load_dbc_fields
from can_discovery.decoding.field_values import extract
from can_discovery.decoding.signals import (
    FieldCandidate,
    _bit_positions,
    discover_fields,
)
from can_discovery.experiments.picacan import (
    Signal,
    aligned,
    bit_flip_fields,
    directed_edge,
    exclude_status_and_counters,
    groups,
    match_ice,
    series,
)
from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical.physics_factors import evaluate_factor

ROLE_WORDS = {
    'accelerator': ('accelerator', 'pedal_gas', 'gas_pedal'),
    'brake': ('brake_pressure', 'brake_pedal', 'brake_position'),
    'throttle': ('throttle_position', 'throttle_angle', 'throttle_plate'),
    'engine_speed': ('engine_rpm', 'engine_speed', 'rpm'),
    'vehicle_speed': ('vehicle_speed', 'car_speed', 'veh_speed'),
}


def truth_role(name):
    lowered = name.lower()
    for role, aliases in ROLE_WORDS.items():
        if any(alias in lowered for alias in aliases):
            return role
    return None


def truth_metrics(signals, semantic, matching, truth, captures):
    """Post-fit boundary and role audit; absent DBC roles stay unassessable."""
    active = []
    for field in truth:
        if field.mux_start is not None:
            continue
        values = [extract(frame.payload, field.start, field.width, field.endian, field.signed)
                  for capture in captures for frame in capture.frames
                  if frame.payload and frame.arb_id.upper() == field.arb_id.upper()
                  and len(frame.payload) == field.payload_length]
        if len(set(values)) > 1:
            active.append(field)
    exact, equivalent, labels = 0, 0, {}
    for index, signal in enumerate(signals):
        same = [field for field in active if field.arb_id.upper() == signal.arb_id.upper()
                and field.payload_length == signal.length]
        bits = _bit_positions(signal.start, signal.width, signal.endian)
        matching_bits = [field for field in same
                         if _bit_positions(field.start, field.width, field.endian) == bits]
        if matching_bits:
            exact += 1
            labels[index] = [field.label_hypotheses[0][0] for field in matching_bits]
            equivalent += 1
            continue
        for field in same:
            if all(all(extract(frame.payload, signal.start, signal.width, signal.endian, False)
                       == extract(frame.payload, field.start, field.width, field.endian,
                                  field.signed)
                       for frame in capture.frames if frame.payload
                       and frame.arb_id.upper() == signal.arb_id.upper()
                       and len(frame.payload) == signal.length) for capture in captures):
                equivalent += 1
                labels[index] = [field.label_hypotheses[0][0]]
                break
    sets = []
    for group in semantic:
        roles = sorted({role for member in group['members']
                        for label in labels.get(member, ())
                        if (role := truth_role(label))})
        pivot_roles = sorted({role for pivot in group['pivots']
                              for label in labels.get(pivot, ())
                              if (role := truth_role(label))})
        sets.append({'roles': roles, 'pivot_roles': pivot_roles,
                     'mixed_known_roles': len(roles) > 1,
                     'known_members': sum(member in labels for member in group['members'])})
    assessable = sorted({role for field in active
                         if (role := truth_role(field.label_hypotheses[0][0]))})
    retrieval = 'unassessable_missing_dbc_truth' if len(assessable) < 5 else 'no_unique_match'
    if matching['status'] == 'unique' and len(assessable) == 5:
        mapped = [sets[index]['roles'] for index in matching['matches'][0]]
        retrieval = 'correct' if all(mapped[i] == [role] for i, role in enumerate(ROLE_WORDS)) else 'incorrect'
    return {'active_dbc_fields': len(active), 'exact_candidates': exact,
            'value_equivalent_candidates': equivalent,
            'labeled_candidates': len(labels), 'assessable_roles': assessable,
            'semantic_sets': sets, 'five_role_retrieval': retrieval}


def checked_captures(records):
    captures = []
    for record in records:
        path = Path(record['capture'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record['converted_sha256']:
            raise ValueError(f'converted hash changed: {path}')
        captures.append(load_capture(path))
    return captures


def signal_from_field(field):
    return Signal(field.arb_id, field.start, field.width, field.endian, field.payload_length)


def estimate_graph(captures, signals, semantic, *, min_observations, min_changes,
                   shift_bins=0):
    """Estimate every retained early-pivot pairing without joining routes."""
    edges = {}
    evidence = []
    trace_cache = {(route, index): series(capture, signal)
                   for route, capture in enumerate(captures)
                   for index, signal in enumerate(signals)
                   if any(index in group['pivots'] for group in semantic)}
    for i in range(len(semantic)):
        for j in range(i+1, len(semantic)):
            pair_votes = []
            candidates = [(left, right) for left in semantic[i]['pivots']
                          for right in semantic[j]['pivots']]
            unavailable = []
            for left, right in candidates:
                aligned_routes = [aligned(trace_cache[route, left], trace_cache[route, right])
                                  for route in range(len(captures))]
                if shift_bins:
                    aligned_routes = [(a[:-shift_bins], b[shift_bins:])
                                      for a, b in aligned_routes if len(a) > shift_bins]
                observations = sum(len(a) for a, _ in aligned_routes)
                changes = (sum(np.count_nonzero(np.diff(a)) for a, _ in aligned_routes),
                           sum(np.count_nonzero(np.diff(b)) for _, b in aligned_routes))
                if observations < min_observations or min(changes) < min_changes or \
                        any(len(a) < 2000 for a, _ in aligned_routes):
                    unavailable.append({'pivots': [left, right],
                                        'reason': 'insufficient_activity',
                                        'observations': observations, 'changes': changes})
                    continue
                votes = [directed_edge(a, b) for a, b in aligned_routes]
                if any(vote is None for vote in votes):
                    unavailable.append({'pivots': [left, right], 'reason': 'unavailable_peak'})
                    continue
                pair_votes.extend(votes)
            if unavailable or not pair_votes:
                evidence.append({'pair': [i, j], 'status': 'unavailable',
                                 'pivot_pairs': len(candidates), 'reasons': unavailable})
                continue
            directions = {vote[:2] for vote in pair_votes}
            signs = {vote[3] for vote in pair_votes}
            if len(directions) != 1 or len(signs) != 1:
                evidence.append({'pair': [i, j], 'status': 'pivot_or_route_disagreement',
                                 'pivot_pairs': len(candidates)})
                continue
            direction = pair_votes[0][:2]
            lag = int(np.median([vote[2] for vote in pair_votes]))
            sign = pair_votes[0][3]
            source, target = (i, j) if direction == (0, 1) else (j, i)
            edges[source, target] = (lag, sign)
            evidence.append({'pair': [i, j], 'status': 'edge', 'source': source,
                             'target': target, 'delay_ms': lag*10, 'sign': sign,
                             'pivot_pairs': len(candidates)})
    return edges, evidence


def fit(entry, *, ablation='baseline', max_ids=None, oracle_signals=None,
        shift_bins=0):
    if entry.get('powertrain') != 'ICE':
        raise ValueError('ICE template requires verified ICE powertrain metadata')
    started = time.perf_counter()
    fitting = checked_captures(entry['windows']['fitting'])
    if oracle_signals is not None:
        signals = list(oracle_signals)
    elif ablation == 'alternate_layout':
        registry = discover_fields(fitting, max_ids=max_ids or 1000000)
        signals = [signal_from_field(f) for f in registry.candidates if f.signed is not None]
    else:
        signals = bit_flip_fields(fitting, max_ids=max_ids)
    signals = list(dict.fromkeys(signals))
    signals, excluded = exclude_status_and_counters(signals, fitting)
    semantic = groups(signals, fitting, coequal=(ablation == 'coequal_pivots'))
    edges, evidence = estimate_graph(fitting, signals, semantic,
                                     min_observations=10000, min_changes=100,
                                     shift_bins=shift_bins)
    matching = match_ice(edges, len(semantic), brake_sign_only=(ablation == 'brake_sign_only'))
    factor_audit = None
    if ablation == 'physical_factors':
        factor_audit = {'status': 'unavailable_nonunique_match'}
        if matching['status'] == 'unique':
            mapping = matching['matches'][0]
            selected = [signals[semantic[node]['pivots'][0]] for node in mapping]
            fields = [FieldCandidate(s.arb_id, s.start, s.width, s.endian, False,
                      'unknown', 0., 0., (), (), (), s.length) for s in selected]
            checks = {}
            for name, positions in {'speed_throttle': (4, 2),
                                    'rpm_throttle': (3, 2),
                                    'speed_brake': (4, 1)}.items():
                evidence = evaluate_factor(fitting, tuple(fields[i] for i in positions), name)
                checks[name] = {'status': evidence.status, 'effect': evidence.effect,
                                'control': evidence.control_effect}
            factor_audit = {'status': 'supported' if all(item['status'] == 'supported'
                             for item in checks.values()) else 'not_supported', 'checks': checks}
            if factor_audit['status'] != 'supported':
                matching = {**matching, 'status': 'physical_factor_abstain'}
    return {'version': 1, 'vehicle': entry['vehicle'], 'ablation': ablation,
            'max_ids': max_ids,
            'shift_bins': shift_bins,
            'grid_ms': 10, 'pearson_lag_ms': [-1000, 1000],
            'ratio_delay_ms': [10, 3000], 'embedding': [6, 20],
            'signals': [s.__dict__ if hasattr(s, '__dict__') else
                        {'arb_id': s.arb_id, 'start': s.start, 'width': s.width,
                         'endian': s.endian, 'length': s.length} for s in signals],
            'semantic_sets': semantic, 'excluded_fields': [
                {'signal': {'arb_id': item['signal'].arb_id,
                            'start': item['signal'].start,
                            'width': item['signal'].width,
                            'endian': item['signal'].endian,
                            'length': item['signal'].length},
                 'reason': item['reason']} for item in excluded],
            'edges': evidence, 'match': matching, 'physical_factor_audit': factor_audit,
            'fitting_hashes': [c.sha256 for c in fitting],
            'fit_seconds': time.perf_counter()-started,
            'promotion_status': 'experimental_unpromoted'}


def assess(result, entry, role):
    captures = checked_captures(entry['windows'][role])
    signals = [Signal(**record) for record in result['signals']]
    activity = []
    for group_index, group in enumerate(result['semantic_sets']):
        for pivot in group['pivots']:
            signal = signals[pivot]
            for capture in captures:
                _, values = series(capture, signal)
                activity.append({'group': group_index, 'pivot': pivot,
                                 'capture_hash': capture.sha256,
                                 'observations': len(values),
                                 'changes': int(np.count_nonzero(np.diff(values))),
                                 'available': len(values) >= 2000 and
                                              np.count_nonzero(np.diff(values)) >= 20})
    route_graphs = []
    for capture in captures:
        edges, evidence = estimate_graph([capture], signals, result['semantic_sets'],
                                         min_observations=2000, min_changes=20,
                                         shift_bins=result.get('shift_bins', 0))
        route_graphs.append({'capture_hash': capture.sha256, 'edges': evidence,
                             'match': match_ice(edges, len(result['semantic_sets']),
                                  brake_sign_only=result['ablation'] == 'brake_sign_only')})
    # The target DBC is opened only after fit and graph assessment have returned.
    dbc = Path(entry['dbc'])
    if hashlib.sha256(dbc.read_bytes()).hexdigest() != entry['dbc_sha256']:
        raise ValueError('DBC hash changed')
    truth = load_dbc_fields(dbc)
    candidates = tuple(FieldCandidate(s.arb_id, s.start, s.width, s.endian, False,
        'unknown', 0., 0., (), (), (), s.length) for s in signals)
    coverage = [evaluate_discovery(candidates, truth, capture) for capture in captures]
    quality = truth_metrics(signals, result['semantic_sets'], result['match'], truth, captures)
    fitted = result['match']
    matched_groups = set(fitted['matches'][0]) if fitted['status'] == 'unique' else set()
    active = all(item['available'] for item in activity
                 if item['group'] in matched_groups)
    status = ('abstain_fit_nonunique_or_incomplete' if fitted['status'] != 'unique'
              else 'abstain_insufficient_activity' if not active
              else 'replicated' if all(graph['match']['status'] == 'unique' and
                  graph['match']['matches'] == fitted['matches'] for graph in route_graphs)
              else 'not_replicated')
    fitting_edges = {(item['source'], item['target'], item['sign'])
                     for item in result['edges'] if item['status'] == 'edge'}
    replication = []
    for graph in route_graphs:
        observed = {(item['source'], item['target'], item['sign'])
                    for item in graph['edges'] if item['status'] == 'edge'}
        replication.append({'replicated': len(fitting_edges & observed),
                            'fitting_edges': len(fitting_edges),
                            'assessment_edges': len(observed)})
    return {'role': role, 'activity': activity, 'field_coverage': coverage,
            'quality': quality, 'directed_edge_replication': replication,
            'route_graphs': route_graphs, 'match_status': status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sealed', action='store_true')
    parser.add_argument('--max-ids', type=int, default=None,
                        help='audit cap; default processes every observed ID')
    parser.add_argument('--shifted-control', action='store_true',
                        help='separate 10-second shifted target control on fitting routes')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('version') != 1:
        raise ValueError('expected frozen PicaCAN manifest v1')
    output = []
    for entry in manifest['vehicles']:
        baseline = fit(entry, max_ids=args.max_ids)
        baseline['development'] = assess(baseline, entry, 'development')
        baseline['ablations'] = {}
        for name in ('alternate_layout', 'coequal_pivots', 'brake_sign_only',
                     'physical_factors'):
            variant = fit(entry, ablation=name, max_ids=args.max_ids)
            variant['development'] = assess(variant, entry, 'development')
            baseline['ablations'][name] = variant
        # This diagnostic alone may read target layouts before its own fit.
        dbc = Path(entry['dbc'])
        if hashlib.sha256(dbc.read_bytes()).hexdigest() != entry['dbc_sha256']:
            raise ValueError('DBC hash changed')
        oracle_fields = load_dbc_fields(dbc)
        oracle_signals = list(dict.fromkeys(signal_from_field(field) for field in oracle_fields
                                           if field.mux_start is None))
        oracle = fit(entry, ablation='dbc_oracle_extraction', max_ids=args.max_ids,
                     oracle_signals=oracle_signals)
        oracle['development'] = assess(oracle, entry, 'development')
        baseline['ablations']['dbc_oracle_extraction'] = oracle
        if args.shifted_control:
            control = fit(entry, ablation='shifted_time_control', max_ids=args.max_ids,
                          shift_bins=1000)
            control['development'] = assess(control, entry, 'development')
            baseline['shifted_time_control'] = control
        if args.sealed:
            baseline['sealed'] = assess(baseline, entry, 'sealed')
        output.append(baseline)
        args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
