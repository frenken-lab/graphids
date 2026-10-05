"""Bounded, null-aware combination search for physical roles."""
from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
from ..decoding.signals import FieldCandidate, _bit_positions
from ..preprocessing.capture import Capture
from .source_calibration import SourceRoleCalibration, SourceFactorValidation
from .evidence import FactorEvidence
from .physics_factors import FACTOR_ROLES
from .observational_edges import GRAPH_EDGES
from .vehicle_models import MODEL_ROLES
from .physical_roles import PhysicalDiscovery

def fit_physical_roles(captures: list[Capture], *, max_ids: int = 64,
                       samples_per_id: int = 512, max_role_candidates: int = 64,
                       max_family_candidates: int = 8,
                       beam_width: int = 64, max_combinations: int = 8,
                       message_ids: tuple[str, ...] | None = None,
                       protocol_fields: tuple[FieldCandidate, ...] | None = None,
                       candidates: tuple[FieldCandidate, ...] | None = None,
                       layout_memberships: dict[str, tuple[str, ...]] | None = None,
                       source_calibration: tuple[SourceRoleCalibration, ...] = (),
                       validated_factors: tuple[SourceFactorValidation, ...] = (),
                       _functions: dict | None = None) -> PhysicalDiscovery:
    """Rank compatible physical-role sets with a null assignment for each role."""
    if _functions is None:
        from . import physical_roles as compatibility
        _functions = vars(compatibility)
    ROLES = _functions['ROLES']
    RUNTIME_ROLES = _functions['RUNTIME_ROLES']
    RoleHypothesis = _functions['RoleHypothesis']
    CandidateCombination = _functions['CandidateCombination']
    PhysicalDiscovery = _functions['PhysicalDiscovery']
    _index_capture = _functions['_index_capture']
    _sample_values = _functions['_sample_values']
    _series_metrics = _functions['_series_metrics']
    _role_score = _functions['_role_score']
    _field_id = _functions['_field_id']
    _equivalence_groups = _functions['_equivalence_groups']
    propose_fields = _functions['propose_fields']
    duplicate_signal_groups = _functions['duplicate_signal_groups']
    evaluate_factor = _functions['evaluate_factor']
    evaluate_graph_edge = _functions['evaluate_graph_edge']
    evaluate_vehicle_model = _functions['evaluate_vehicle_model']
    if not (1 <= max_role_candidates <= 64 and 1 <= max_family_candidates <= 8):
        raise ValueError('physical candidate budgets must be within 64 per role and 8 per family')
    all_candidates = (candidates if candidates is not None else
                      propose_fields(captures, max_ids=max_ids,
                                     samples_per_id=samples_per_id,
                                     max_candidates_per_id=16384,
                                     only_ids=message_ids,
                                     protocol_fields=protocol_fields))
    protocol = tuple(field for field in all_candidates if field.role in ("counter", "checksum"))
    reserved = {(field.bus, field.arb_id, field.payload_length): set()
                for field in protocol}
    for field in protocol:
        reserved[(field.bus, field.arb_id, field.payload_length)].update(
            _bit_positions(field.start, field.width, field.endian))
    by_family: dict[tuple[str | None, str, int], list[FieldCandidate]] = {}
    for field in all_candidates:
        if field.role == "unknown" and field.signed is not None:
            if (_bit_positions(field.start, field.width, field.endian) &
                    reserved.get((field.bus, field.arb_id, field.payload_length), set())):
                continue
            by_family.setdefault((field.bus, field.arb_id, field.payload_length), []).append(field)
    indices = [_index_capture(capture) for capture in captures]
    hypotheses: list[RoleHypothesis] = []
    rejected = []
    calibrators = {item.role: item for item in source_calibration}
    if source_calibration:
        rejected.extend({'role': role, 'reason': 'source_calibration_unavailable',
                         'fallback': 'physics_only'}
                        for role in ROLES if role not in calibrators)
    # Stream the full proposed lattice. Holding every decoded trace would cost
    # gigabytes on the pinned captures; keep only the best field per trace.
    pools: dict[str, dict[tuple, RoleHypothesis]] = {role: {} for role in ROLES}
    threshold_counts = {role: 0 for role in ROLES}
    for family, fields in by_family.items():
        for field in fields:
            history = [_sample_values(capture, field, cap=256, index=index)
                       for capture, index in zip(captures, indices, strict=True)]
            metrics = _series_metrics(history)
            if metrics is None:
                continue
            signature = []
            for _, values in history:
                centered = values - values[0]
                scale = np.max(np.abs(centered))
                signature.append(np.round(centered / scale, 5).astype(np.float32).tobytes()
                                 if scale else b'constant')
            key = (family, field.mux_start, field.mux_value, *signature)
            support = tuple(len(values) for _, values in history)
            for role in ROLES:
                raw_score, evidence = _role_score(role, history, field, metrics)
                calibrator = calibrators.get(role)
                threshold = calibrator.inclusion_threshold if calibrator else .30
                if not math.isfinite(raw_score) or raw_score < threshold:
                    continue
                threshold_counts[role] += 1
                score = raw_score + (calibrator.adjustment(raw_score) if calibrator else 0.)
                item = RoleHypothesis(field, role, score, support,
                                      evidence + (("source_dbc_calibrated",) if calibrator else ()),
                                      calibration="source_dbc" if calibrator else "physics_only")
                prior = pools[role].get(key)
                if prior is None or (score, field.width) > (prior.unary_score, prior.candidate.width):
                    pools[role][key] = item
    for role in ROLES:
        calibrator = calibrators.get(role)
        threshold = calibrator.inclusion_threshold if calibrator else .30
        rejected.append({'role': role, 'reason': 'pool_threshold', 'threshold': threshold,
                         'method': 'source_positive_p05' if calibrator else 'broad_physics_fallback',
                         'passing_layouts': threshold_counts[role],
                         'distinct_traces': len(pools[role]),
                         'alias_grouped': threshold_counts[role] - len(pools[role])})
        ranked = sorted(pools[role].values(), key=lambda item: (-item.unary_score,
                                                                  _field_id(item.candidate)))
        per_family: dict[tuple[str, int], int] = {}
        selected_count = 0
        for item in ranked:
            family = (item.candidate.bus, item.candidate.arb_id,
                      item.candidate.payload_length)
            if selected_count >= max_role_candidates or \
                    per_family.get(family, 0) >= max_family_candidates:
                rejected.append({'role': role, 'reason': 'candidate_budget_overflow',
                                 'field': _field_id(item.candidate), 'score': item.unary_score,
                                 'threshold': threshold})
                continue
            hypotheses.append(item)
            selected_count += 1
            per_family[family] = per_family.get(family, 0) + 1
        if not per_family:
            rejected.append({'role': role, 'reason': 'no_supported_candidate'})
    groups = _equivalence_groups(hypotheses, captures, indices)
    graph_enabled = any(item.name.startswith('graph:') and
                        len(item.supporting_vehicles) >= 2 for item in validated_factors)
    if graph_enabled:
        groups, early_copies = duplicate_signal_groups(hypotheses,captures,groups)
    else:
        early_copies = set(range(len(hypotheses)))
    calibrated = []
    for number,(item, group) in enumerate(zip(hypotheses, groups, strict=True)):
        histories = [_sample_values(capture, item.candidate, cap=8192, index=index)[1]
                     for capture, index in zip(captures, indices, strict=True)]
        differences = [np.abs(np.diff(values)) for values in histories if len(values) > 1]
        limit = max(1.0, *(float(np.quantile(delta, .995) * 1.5) for delta in differences))
        states = tuple(sorted({int(value) for values in histories for value in values}))
        allowed = (states if item.role == "gear" and len(states) <= 32 and
                   any(len(np.unique(values)) > 1 for values in histories) else ())
        pivot = number in early_copies
        calibrated.append(RoleHypothesis(item.candidate, item.role,
                                         item.unary_score + (.01 if graph_enabled and pivot else 0.),
                                         item.support,
                                         item.evidence + (('earliest_duplicate_candidate',)
                                                          if graph_enabled and pivot else ()),
                                         group, limit, allowed,
                                         item.calibration))
    hypotheses = calibrated
    by_role = {role: [i for i, item in enumerate(hypotheses) if item.role == role]
               for role in ROLES}
    def compatible_layout(index: int, assignments: tuple[int | None, ...]) -> bool:
        if layout_memberships is None:
            return True
        from .candidate_graph import field_key
        field = hypotheses[index].candidate
        family = (field.bus, field.arb_id, field.payload_length,
                  field.mux_start, field.mux_value)
        options = set(layout_memberships.get(field_key(field), ()))
        if not options:
            return False
        for previous in assignments:
            if previous is None:
                continue
            other = hypotheses[previous].candidate
            other_family = (other.bus, other.arb_id, other.payload_length,
                            other.mux_start, other.mux_value)
            if other_family == family:
                options &= set(layout_memberships.get(field_key(other), ()))
                if not options:
                    return False
        return True
    # Unary search keeps the best completion for every newly introduced role
    # candidate, so a later factor can rescue a weak unary rank.
    beam: list[tuple[float, tuple[int | None, ...]]] = [(0.0, ())]
    for role in ROLES:
        next_beam = []
        for score, assignments in beam:
            for index in (None, *by_role[role]):
                if index is None:
                    next_beam.append((score, assignments + (None,)))
                    continue
                field = hypotheses[index].candidate
                if not compatible_layout(index, assignments):
                    continue
                if any(previous is not None and
                       hypotheses[previous].candidate.arb_id == field.arb_id and
                       hypotheses[previous].candidate.bus == field.bus and
                       hypotheses[previous].candidate.payload_length == field.payload_length and
                       hypotheses[previous].candidate.mux_value == field.mux_value and
                       _bit_positions(hypotheses[previous].candidate.start,
                                      hypotheses[previous].candidate.width,
                                      hypotheses[previous].candidate.endian) &
                       _bit_positions(field.start, field.width, field.endian)
                       for previous in assignments):
                    continue
                total = score + hypotheses[index].unary_score - .45
                next_beam.append((total, assignments + (index,)))
        next_beam.sort(key=lambda item: (-item[0], str(item[1])))
        reserved = {}
        for item in next_beam:
            for position, candidate in enumerate(item[1]):
                reserved.setdefault((position, candidate), item)
        beam = sorted({item[1]: item for item in (*next_beam[:beam_width], *reserved.values())}.values(),
                      key=lambda item: (-item[0], str(item[1])))[:max(beam_width, len(reserved))]
    # Evaluate every completed assignment against every applicable factor.
    # Missing roles leave a factor unavailable, not failed.
    factor_cache: dict[tuple, FactorEvidence] = {}
    validated_names = {item.name for item in validated_factors
                       if len(item.supporting_vehicles) >= 2}
    def complete_score(assignments: tuple[int | None, ...]):
        unary = sum(hypotheses[index].unary_score - .45
                    for index in assignments if index is not None)
        evidence = []
        for name, roles in FACTOR_ROLES.items():
            positions = [ROLES.index(role) for role in roles if role in ROLES]
            if len(positions) != len(roles):
                continue
            if any(assignments[pos] is None for pos in positions):
                evidence.append(FactorEvidence(name, roles, 'unavailable', reason='missing_role'))
                continue
            members = tuple(assignments[pos] for pos in positions)
            key = (name, *members)
            if key not in factor_cache:
                factor_cache[key] = replace(evaluate_factor(captures,
                    tuple(hypotheses[index].candidate for index in members), name, indices),
                    members=members)
            evidence.append(factor_cache[key])
        for name, (source_role, target_role, _) in GRAPH_EDGES.items():
            if 'graph:'+name not in validated_names:
                continue
            source_position, target_position = ROLES.index(source_role), ROLES.index(target_role)
            if assignments[source_position] is None or assignments[target_position] is None:
                continue
            members = (assignments[source_position],assignments[target_position])
            key = ('graph:'+name,*members)
            if key not in factor_cache:
                result = evaluate_graph_edge(captures,
                    hypotheses[members[0]].candidate,hypotheses[members[1]].candidate,name)
                factor_cache[key] = FactorEvidence('graph:'+name,(source_role,target_role),
                    result.status,result.gain,result.control_gain,
                    int(round(result.lag_seconds/.02)),0,min(result.support,default=0),
                    result.reason,members=members)
            evidence.append(factor_cache[key])
        for name,roles in MODEL_ROLES.items():
            positions = [ROLES.index(role) for role in roles]
            if any(assignments[pos] is None for pos in positions):
                continue
            if name != 'model:dynamic_bicycle' and name not in validated_names:
                continue
            members = tuple(assignments[pos] for pos in positions)
            key = (name,*members)
            if key not in factor_cache:
                factor_cache[key] = replace(evaluate_vehicle_model(captures,
                    tuple(hypotheses[index].candidate for index in members),name),
                    members=members)
            evidence.append(factor_cache[key])
        total = unary + sum((.15 if item.name.startswith(('graph:','model:')) else .35)*item.effect
                            for item in evidence if item.status == 'supported')
        return (total, unary, assignments, tuple(evidence))
    scored = [complete_score(assignments) for _, assignments in beam]
    scored.sort(key=lambda item: (-item[0], str(item[2])))
    # Coordinate refinement gives every threshold-pool member a chance to
    # replace a role in a strong completed graph. All replacements are scored
    # against the full factor set, including the three-field gear relation.
    visited = {item[2] for item in scored}
    seeds = scored[:min(2, len(scored))]
    for seed in seeds:
        current = seed
        for _ in range(2):
            neighbors = []
            for position, role in enumerate(ROLES):
                for candidate in (None, *by_role[role]):
                    if candidate == current[2][position]:
                        continue
                    assignment = current[2][:position] + (candidate,) + current[2][position + 1:]
                    if assignment in visited:
                        continue
                    if candidate is not None:
                        if not compatible_layout(candidate, assignment):
                            continue
                        field = hypotheses[candidate].candidate
                        bits = _bit_positions(field.start, field.width, field.endian)
                        if any(other is not None and offset != position and
                               hypotheses[other].candidate.arb_id == field.arb_id and
                               hypotheses[other].candidate.bus == field.bus and
                               hypotheses[other].candidate.payload_length == field.payload_length and
                               hypotheses[other].candidate.mux_value == field.mux_value and
                               bits & _bit_positions(hypotheses[other].candidate.start,
                                                     hypotheses[other].candidate.width,
                                                     hypotheses[other].candidate.endian)
                               for offset, other in enumerate(assignment)):
                            continue
                    visited.add(assignment)
                    neighbors.append(complete_score(assignment))
            if not neighbors:
                break
            scored.extend(neighbors)
            winner = max(neighbors, key=lambda item: item[0])
            if winner[0] <= current[0]:
                break
            current = winner
    scored.sort(key=lambda item: (-item[0], str(item[2])))
    null_alternatives = []
    if scored:
        best = scored[0]
        for pos, role in enumerate(ROLES):
            index = best[2][pos]
            if index is None:
                continue
            remaining = tuple(factor for factor in best[3] if role not in factor.roles)
            unary = best[1] - (hypotheses[index].unary_score - .45)
            total = unary + sum((.15 if factor.name.startswith(('graph:','model:')) else .35)*factor.effect
                                for factor in remaining if factor.status == 'supported')
            assignment = best[2][:pos] + (None,) + best[2][pos + 1:]
            null_alternatives.append((total, unary, assignment, remaining))
    selected = []
    seen = set()
    for item in scored:
        signature = tuple(hypotheses[index].equivalence_group if index is not None else None
                          for index in item[2])
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(item)
        if len(selected) >= max(1, max_combinations - min(2, len(null_alternatives))):
            break
    null_alternatives.sort(key=lambda item: -item[0])
    for alternative in null_alternatives:
        if len(selected) >= max_combinations:
            break
        if alternative[2] not in (item[2] for item in selected):
            selected.append(alternative)
    selected.sort(key=lambda item: -item[0])
    weights = np.exp(np.asarray([item[0] for item in selected]) - selected[0][0])
    weights /= weights.sum()
    active_weights = {item.name: item.weight for item in validated_factors
                      if len(item.supporting_vehicles) >= 2}
    runtime_scores = [sum(hypotheses[index].unary_score - .45
                          for role, index in zip(ROLES, item[2])
                          if role in RUNTIME_ROLES and index is not None) +
                      sum(active_weights.get(factor.name, 0.) * factor.effect
                                   for factor in item[3] if factor.status == 'supported')
                      for item in selected]
    runtime = np.exp(np.asarray(runtime_scores) - max(runtime_scores))
    runtime /= runtime.sum()
    combinations = tuple(CandidateCombination(assignments, float(weight), float(score),
                         tuple((factor.name, factor.effect) for factor in evidence), evidence,
                         float(live_weight))
                         for (score, _, assignments, evidence), weight, live_weight in
                         zip(selected, weights, runtime, strict=True))
    # Redundant speeds form an equivalence check, not a second required role.
    speed_candidates = by_role['speed'][:8]
    for left in range(len(speed_candidates)):
        for right in range(left + 1, len(speed_candidates)):
            first, second = speed_candidates[left], speed_candidates[right]
            if hypotheses[first].candidate.arb_id == hypotheses[second].candidate.arb_id:
                continue
            factor_cache[('wheel_speed', first, second)] = replace(evaluate_factor(
                captures, (hypotheses[first].candidate, hypotheses[second].candidate),
                'wheel_speed', indices), members=(first, second))
    return PhysicalDiscovery(tuple(hypotheses), combinations, protocol, tuple(rejected),
                             tuple(factor_cache.values()))
