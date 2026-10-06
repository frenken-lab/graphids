"""Audit ranked physical CAN hypotheses with held-out recordings and an optional DBC.

The target DBC is opened only after role fitting; source calibration, when
requested, reads DBCs from other vehicles listed in a JSON manifest.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from can_discovery.decoding.dbc import load_dbc_fields, rank_dbc
from can_discovery.decoding.field_values import extract
from can_discovery.decoding.signals import (
    _bit_positions,
    discover_fields,
    propose_fields,
)
from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical import (
    ROLES,
    _dbc_role,
    _role_score,
    _sample_values,
    _series_metrics,
    fit_physical_roles,
    fit_source_calibration,
    fit_source_factor_validation,
)
from can_discovery.relationships.physical.candidate_graph import (
    GraphWeights,
    layout_memberships,
    rerank,
    selected_layout_fields,
)
from can_discovery.relationships.physical.observational_edges import (
    GRAPH_EDGES,
    evaluate_graph_edge,
)
from can_discovery.relationships.physical.physics_factors import (
    FACTOR_ROLES,
    evaluate_factor,
)
from can_discovery.relationships.physical.vehicle_models import (
    MODEL_ROLES,
    evaluate_vehicle_model,
)


def _field_key(field):
    return (field.arb_id, field.payload_length, field.start, field.width,
            field.endian, field.signed, field.mux_start, field.mux_value)


def _paired_values(candidate, truth, capture) -> np.ndarray:
    pairs = []
    for frame in capture.frames:
        if frame.arb_id != truth.arb_id or frame.payload is None or len(frame.payload) != truth.payload_length:
            continue
        if truth.mux_start is not None and frame.payload[truth.mux_start // 8] != truth.mux_value:
            continue
        if candidate.mux_start is not None and frame.payload[candidate.mux_start // 8] != candidate.mux_value:
            continue
        pairs.append((extract(frame.payload, candidate.start, candidate.width,
                              candidate.endian, candidate.signed),
                      extract(frame.payload, truth.start, truth.width,
                              truth.endian, truth.signed)))
    return np.asarray(pairs, dtype=float).reshape(-1, 2)


def _value_metrics(candidate, truth, references, held_out) -> dict | None:
    if (candidate.arb_id, candidate.payload_length) != (truth.arb_id, truth.payload_length):
        return None
    train = np.concatenate([_paired_values(candidate, truth, capture)
                            for capture in references])
    test = _paired_values(candidate, truth, held_out)
    if len(train) < 30 or len(test) < 30:
        return None
    x, y = train.T
    xt, yt = test.T
    if np.ptp(x) == 0 or np.ptp(y) == 0 or np.ptp(yt) == 0:
        return None
    slope, offset = np.polyfit(x, y, 1)
    residual = np.abs(yt - (slope * xt + offset))
    span = float(np.ptp(yt))
    return {'slope': float(slope), 'offset': float(offset),
            'held_out_mean_absolute_raw_error': float(np.mean(residual)),
            'held_out_p99_absolute_raw_error': float(np.quantile(residual, .99)),
            'held_out_relative_mae': float(np.mean(residual) / span),
            'value_equivalent': bool(slope > 0 and np.quantile(residual, .99) <=
                                     max(1.0, .01 * span))}


def _value_equivalence(candidate, truth, references, held_out) -> bool:
    metrics = _value_metrics(candidate, truth, references, held_out)
    return bool(metrics and metrics['value_equivalent'])


def _active(field, capture) -> bool:
    values = set()
    for frame in capture.frames:
        if frame.arb_id != field.arb_id or frame.payload is None or len(frame.payload) != field.payload_length:
            continue
        if field.mux_start is not None and frame.payload[field.mux_start // 8] != field.mux_value:
            continue
        values.add(extract(frame.payload, field.start, field.width, field.endian, field.signed))
        if len(values) > 1:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference_a', type=Path)
    parser.add_argument('reference_b', type=Path)
    parser.add_argument('held_out', type=Path)
    parser.add_argument('--max-ids', type=int, default=64)
    parser.add_argument('--target-vehicle', default='unknown')
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--source-benchmark-manifest', type=Path)
    parser.add_argument('--evaluation-dbc', type=Path)
    parser.add_argument('--audit-candidate-pool', action='store_true')
    parser.add_argument('--graph-artifact', type=Path)
    parser.add_argument('--max-layouts', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    references = [load_capture(path) for path in (args.reference_a, args.reference_b)]
    held_out = load_capture(args.held_out)
    if len({item.sha256 for item in (*references, held_out)}) != 3:
        raise SystemExit('reference and held-out captures must be distinct')
    source_calibration = ()
    validated_factors = ()
    if args.source_manifest:
        entries = json.loads(args.source_manifest.read_text())
        entries = [entry for entry in entries
                   if entry['vehicle'].casefold() != args.target_vehicle.casefold()]
        sources = [(entry['vehicle'], [load_capture(Path(path)) for path in entry['captures']],
                    Path(entry['dbc'])) for entry in entries]
        source_calibration = fit_source_calibration(sources, target_vehicle=args.target_vehicle)
    if args.source_benchmark_manifest:
        entries = json.loads(args.source_benchmark_manifest.read_text())
        entries = [entry for entry in entries
                   if entry['vehicle'].casefold() != args.target_vehicle.casefold()]
        sources = [(entry['vehicle'],
                    [load_capture(Path(path)) for path in entry.get('fitting', entry.get('references'))],
                    [load_capture(Path(path)) for path in entry.get('validation', [entry['held_out']] if 'held_out' in entry else ())],
                    Path(entry['dbc']))
                   for entry in entries]
        validated_factors = fit_source_factor_validation(
            sources, target_vehicle=args.target_vehicle)
    started = time.perf_counter()
    registry = discover_fields(references, max_ids=args.max_ids,
                               max_layouts=args.max_layouts)
    memberships = (layout_memberships(registry.candidates, registry.tokenizations)
                   if args.graph_artifact else None)
    discovery = fit_physical_roles(references, max_ids=args.max_ids,
                                   source_calibration=source_calibration,
                                   validated_factors=validated_factors,
                                   candidates=registry.candidates if args.graph_artifact else None,
                                   layout_memberships=memberships)
    graph = None
    if args.graph_artifact:
        artifact = GraphWeights.read(args.graph_artifact)
        artifact.check_target(args.target_vehicle, (*references, held_out))
        graph = rerank(discovery, memberships, artifact)
        discovery = replace(discovery, combinations=graph.combinations)
    result = {'seconds': time.perf_counter() - started,
              'peak_memory_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'max_ids': args.max_ids,
              'references': [str(args.reference_a), str(args.reference_b)],
              'held_out': str(args.held_out),
              'calibration': [asdict(item) for item in source_calibration],
              'source_factor_validation': [asdict(item) for item in validated_factors],
              'factor_evidence': [asdict(item) for item in discovery.factor_evidence],
              'hypotheses': [asdict(item) for item in discovery.hypotheses],
              'combinations': [asdict(item) for item in discovery.combinations],
              'rejected': discovery.rejected}
    if graph:
        result['candidate_graph'] = {'nodes': [asdict(item) for item in graph.nodes],
            'evidence': [asdict(item) for item in graph.evidence],
            'layout_choices': graph.layout_choices,
            'artifact': str(args.graph_artifact), 'max_layouts': args.max_layouts}
    held_out_relations = []
    relation_cache = {}
    for rank, combination in enumerate(discovery.combinations, 1):
        for name, fit_effect in combination.relation_evidence:
            fit_factor = next(factor for factor in combination.factors if factor.name == name)
            roles = (FACTOR_ROLES.get(name) or MODEL_ROLES.get(name) or
                     GRAPH_EDGES.get(name.removeprefix('graph:'), ())[:2])
            if not roles:
                continue
            indices = tuple(combination.assignments[ROLES.index(role)] for role in roles)
            if any(index is None for index in indices):
                continue
            key = (name, *indices)
            if key not in relation_cache:
                fields = tuple(discovery.hypotheses[index].candidate for index in indices)
                if name in FACTOR_ROLES:
                    relation_cache[key] = evaluate_factor([held_out], fields, name,
                        forced_lag=fit_factor.lag_bins)
                elif name in MODEL_ROLES:
                    relation_cache[key] = evaluate_vehicle_model([held_out], fields, name,
                        forced_lag=fit_factor.lag_bins)
                else:
                    edge = evaluate_graph_edge([held_out], fields[0], fields[1],
                        name.removeprefix('graph:'),
                        forced_lag_seconds=fit_factor.lag_bins * .02)
                    from can_discovery.relationships.shared.evidence import (
                        FactorEvidence,
                    )
                    relation_cache[key] = FactorEvidence(name, tuple(roles), edge.status,
                        effect=edge.gain, reason=edge.reason)
            held_factor = relation_cache[key]
            held_out_relations.append({'combination_rank': rank, 'relation': name,
                                       'fit_effect': fit_effect,
                                       'fit_status': fit_factor.status,
                                       'fit_direction': fit_factor.direction,
                                       'fit_lag_bins': fit_factor.lag_bins,
                                       'held_out_effect': held_factor.effect,
                                       'held_out_status': held_factor.status,
                                       'held_out_direction': held_factor.direction,
                                       'held_out_lag_bins': held_factor.lag_bins,
                                       'direction_agrees': fit_factor.direction == held_factor.direction,
                                       'lag_agrees': abs(fit_factor.lag_bins - held_factor.lag_bins) <= 3,
                                       'retained': fit_factor.status == 'supported' and
                                                   held_factor.status == 'supported' and
                                                   fit_factor.direction == held_factor.direction and
                                                   abs(fit_factor.lag_bins - held_factor.lag_bins) <= 3})
    result['held_out_relationships'] = held_out_relations
    result['held_out_wheel_equivalence'] = [
        {'members': factor.members, 'fit': asdict(factor),
         'held_out': asdict(evaluate_factor([held_out],
             tuple(discovery.hypotheses[index].candidate for index in factor.members),
             'wheel_speed'))}
        for factor in discovery.factor_evidence
        if factor.name == 'wheel_speed' and factor.members
    ]
    if args.evaluation_dbc:
        fits = [rank_dbc(capture, args.evaluation_dbc)
                for capture in (*references, held_out)]
        if any(fit['coverage'] < .5 or
               fit['matched_length_frames'] / max(1, fit['matched_id_frames']) < .8 or
               fit['decoded_frames'] / max(1, fit['matched_length_frames']) < .99
               for fit in fits):
            raise SystemExit('target vehicle/bus/length does not fit evaluation DBC sufficiently')
        truth = [field for field in load_dbc_fields(args.evaluation_dbc)
                 if _dbc_role(field.label_hypotheses[0][0]) and _active(field, held_out)]
        active_all = [field for field in load_dbc_fields(args.evaluation_dbc)
                      if field.mux_start is None and _active(field, held_out)]
        lattice = (registry.candidates if args.graph_artifact else
                   propose_fields(references, max_ids=args.max_ids,
                                  max_candidates_per_id=16384))
        lattice_keys = {_field_key(field) for field in lattice}
        roles = {}
        truth_diagnostics = []
        selected_keys = {_field_key(item) for item in registry.selected}
        hypothesis_roles = {}
        for item in discovery.hypotheses:
            hypothesis_roles.setdefault(_field_key(item.candidate), set()).add(item.role)
        retrieved_roles = {}
        if discovery.combinations:
            for assigned_role, index in zip(ROLES, discovery.combinations[0].assignments):
                if index is not None:
                    retrieved_roles.setdefault(_field_key(discovery.hypotheses[index].candidate),
                                               set()).add(assigned_role)
        for field in load_dbc_fields(args.evaluation_dbc):
            role = _dbc_role(field.label_hypotheses[0][0])
            if role is None:
                continue
            key = _field_key(field)
            active = _active(field, held_out)
            equivalent = any(_value_equivalence(item.candidate, field, references, held_out)
                             for item in discovery.hypotheses
                             if item.candidate.arb_id == field.arb_id and
                             item.candidate.payload_length == field.payload_length) if active and key not in lattice_keys else False
            if not active:
                status = 'insufficiently_active'
            elif key not in lattice_keys and not equivalent:
                status = 'absent_from_candidate_pool'
            elif key not in lattice_keys:
                status = 'value_equivalent_ambiguous_boundaries'
            elif key in retrieved_roles and role not in retrieved_roles[key]:
                status = 'retrieved_wrong_role'
            elif role not in retrieved_roles.get(key, ()):
                status = 'present_unselected'
            else:
                status = 'retrieved'
            truth_diagnostics.append({'dbc_signal': field.label_hypotheses[0][0],
                                      'role': role, 'status': status,
                                      'active': active, 'candidate_exact': key in lattice_keys,
                                      'selected_layout': key in selected_keys,
                                      'hypothesis_roles': sorted(hypothesis_roles.get(key, ())),
                                      'retrieved_roles': sorted(retrieved_roles.get(key, ()))})
        for role in ('speed', 'rpm', 'gear', 'acceleration', 'steering', 'brake', 'throttle'):
            expected = [field for field in truth if _dbc_role(field.label_hypotheses[0][0]) == role]
            proposed = [item for item in discovery.hypotheses if item.role == role]
            exact = [index + 1 for index, item in enumerate(proposed)
                     if any(_field_key(item.candidate) == _field_key(field) for field in expected)]
            errors = [(index + 1, field.label_hypotheses[0][0], metrics)
                      for index, item in enumerate(proposed)
                      for field in expected
                      if (metrics := _value_metrics(item.candidate, field,
                                                    references, held_out)) is not None]
            equivalent = [rank for rank, _, metrics in errors if metrics['value_equivalent']]
            exact_keys = {_field_key(field) for field in expected}
            equivalent_keys = {_field_key(proposed[rank - 1].candidate)
                               for rank in equivalent}
            combo_exact = []
            combo_equivalent = []
            for rank, combination in enumerate(discovery.combinations, 1):
                index = combination.assignments[ROLES.index(role)]
                if index is None:
                    continue
                key = _field_key(discovery.hypotheses[index].candidate)
                if key in exact_keys:
                    combo_exact.append(rank)
                if key in equivalent_keys:
                    combo_equivalent.append(rank)
            best_error = (min(errors, key=lambda item: item[2]['held_out_relative_mae'])
                          if errors else None)
            roles[role] = {'target_dbc_fields': [field.label_hypotheses[0][0]
                                                  for field in expected],
                           'first_exact_rank': min(exact, default=None),
                           'first_value_equivalent_rank': min(equivalent, default=None),
                           'first_exact_combination_rank': min(combo_exact, default=None),
                           'first_value_equivalent_combination_rank': min(combo_equivalent, default=None),
                           'best_held_out_value_error': (
                               {'rank': best_error[0], 'dbc_signal': best_error[1],
                                **best_error[2]} if best_error else None)}
        result['evaluation'] = {'dbc_fit': fits, 'roles': roles,
                                'truth_diagnostics': truth_diagnostics,
                                'factor_availability': [{'name': item.name,
                                    'status': item.status, 'reason': item.reason,
                                    'lag_bins': item.lag_bins, 'direction': item.direction}
                                    for item in discovery.factor_evidence],
                                'source_factor_support': [asdict(item) for item in validated_factors],
                                'source_factor_availability': [
                                    {'name': name, 'validated': name in {item.name for item in validated_factors},
                                     'source_vehicles': next((item.supporting_vehicles for item in validated_factors
                                                              if item.name == name), ()),
                                     'weight': next((item.weight for item in validated_factors
                                                     if item.name == name), 0.0)}
                                    for name in (*FACTOR_ROLES, *MODEL_ROLES,
                                                 *(f'graph:{name}' for name in GRAPH_EDGES))],
                                'graph_family_weights': dict(artifact.weights) if graph else None,
                                'target_dbc_used_for_fitting': False,
                                'value_equivalence_method':
                                    'affine_map_fit_on_references_tested_on_held_out',
                                'matched_id_length_agreement': [
                                    fit['matched_length_frames'] / max(1, fit['matched_id_frames'])
                                    for fit in fits]}
        calibrators = {item.role: item for item in source_calibration}
        threshold_truth = []
        for field in truth:
            role = _dbc_role(field.label_hypotheses[0][0])
            history = [_sample_values(capture, field, cap=256) for capture in references]
            metrics = _series_metrics(history)
            score = _role_score(role, history, field, metrics)[0] if metrics else float('-inf')
            threshold = calibrators[role].inclusion_threshold if role in calibrators else .30
            threshold_truth.append({'field': field.label_hypotheses[0][0], 'role': role,
                                    'eligible': bool(np.isfinite(score) and score >= threshold and
                                                     _field_key(field) in lattice_keys),
                                    'lattice_retained': _field_key(field) in lattice_keys,
                                    'score': score if np.isfinite(score) else None,
                                    'threshold': threshold})
        result['evaluation']['threshold_pool'] = {
            'known_positive_count': len(threshold_truth),
            'known_positive_retained': sum(item['eligible'] for item in threshold_truth),
            'recall': sum(item['eligible'] for item in threshold_truth) / max(1, len(threshold_truth)),
            'fields': threshold_truth}
        layout_fields = selected_layout_fields(registry.candidates, registry.selected,
            memberships or {}, graph.layout_choices[0]['layouts'] if graph else (),
            discovery.protocol_fields)
        selected = {_field_key(item) for item in layout_fields}
        active_keys = {_field_key(item) for item in active_all}
        exact_selected = len(selected & active_keys)
        result['evaluation']['selected_layout'] = {
            'selected_count': len(selected), 'active_dbc_signals': len(active_keys),
            'exact_count': exact_selected,
            'exact_precision': exact_selected / max(1, len(selected)),
            'active_recall': exact_selected / max(1, len(active_keys))}
        selected_fields = list(layout_fields)
        cross_byte = [field for field in active_all if len({bit // 8 for bit in
                      _bit_positions(field.start, field.width, field.endian)}) > 1]
        order_assessed = [(truth_field, [candidate for candidate in selected_fields
                          if candidate.arb_id == truth_field.arb_id and
                          candidate.payload_length == truth_field.payload_length and
                          _bit_positions(candidate.start, candidate.width, candidate.endian) ==
                          _bit_positions(truth_field.start, truth_field.width, truth_field.endian)])
                          for truth_field in cross_byte]
        order_assessed = [(truth_field, candidates) for truth_field, candidates in
                          order_assessed if candidates]
        result['evaluation']['cross_byte_order'] = {
            'assessed': len(order_assessed),
            'correct': sum(any(candidate.endian == truth_field.endian for candidate in candidates)
                           for truth_field, candidates in order_assessed),
            'accuracy': (sum(any(candidate.endian == truth_field.endian for candidate in candidates)
                             for truth_field, candidates in order_assessed) /
                         len(order_assessed) if order_assessed else None)}
        result['evaluation']['abstentions'] = sum(index is None for index in
            discovery.combinations[0].assignments)
        result['evaluation']['null_roles'] = [role for role, index in
            zip(ROLES, discovery.combinations[0].assignments) if index is None]
        truth_by_role = {role: {_field_key(field) for field in truth
                                if _dbc_role(field.label_hypotheses[0][0]) == role}
                         for role in ROLES}
        def assignment_matches(combination):
            return sum(_field_key(discovery.hypotheses[index].candidate) in truth_by_role[role]
                       for role, index in zip(ROLES, combination.assignments)
                       if index is not None)
        def factor_part(combo):
            return .35 * sum(factor.effect for factor in combo.factors
                             if factor.status == 'supported')
        scores = {'unary_only': lambda combo: combo.score - factor_part(combo),
                  'factors_only': factor_part,
                  'joint_graph': lambda combo: combo.score,
                  'validated_transfer_runtime': lambda combo: combo.runtime_weight or 0.}
        for name in FACTOR_ROLES:
            scores[f'without_{name}'] = lambda combo, name=name: combo.score - .35 * sum(
                factor.effect for factor in combo.factors if factor.name == name and
                factor.status == 'supported')
        ablations = {}
        for name, score_fn in scores.items():
            winner = max(discovery.combinations, key=score_fn)
            ablations[name] = {'top_assignment_exact_roles': assignment_matches(winner),
                               'top_assignment_score': score_fn(winner)}
        result['evaluation']['ablations'] = ablations
        if args.audit_candidate_pool:
            active = active_all
            exact = [field for field in active if _field_key(field) in lattice_keys]
            result['evaluation']['candidate_pool'] = {
                'candidates': len(lattice), 'active_dbc_signals': len(active),
                'exact_layouts_retained': len(exact),
                'active_recall': len(exact) / max(1, len(active)),
                'unretained': [field.label_hypotheses[0][0] for field in active
                               if _field_key(field) not in lattice_keys]}
    content = json.dumps(result, indent=2, allow_nan=False, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + '\n')
    else:
        print(content)


if __name__ == '__main__':
    main()
