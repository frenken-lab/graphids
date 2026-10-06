"""Experimental, bounded evidence graph for physical-role combinations.

The graph contains only candidate interpretations. It never decodes a target DBC.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ...decoding.signals import FieldCandidate, _bit_positions
from ..shared.field_analysis import field_key
from .role_types import ROLES, CandidateCombination, PhysicalDiscovery

VSS_VERSION = 'v6.0'
VSS_SOURCE = 'https://github.com/COVESA/vehicle_signal_specification/tree/v6.0/spec'
VSS_ROLES = {
    'speed': ('Vehicle.Speed', 'float'),
    'rpm': ('Vehicle.Powertrain.CombustionEngine.Speed', 'float'),
    'gear': ('Vehicle.Powertrain.Transmission.CurrentGear', 'int8'),
    'acceleration': ('Vehicle.Acceleration.Longitudinal', 'float'),
    'steering': ('Vehicle.Chassis.SteeringWheel.Angle', 'int16'),
    'brake': ('Vehicle.Chassis.Brake.PedalPosition', 'uint8'),
    'throttle': ('Vehicle.Chassis.Accelerator.PedalPosition', 'uint8'),
    'yaw': ('Vehicle.AngularVelocity.Yaw', 'float'),
    'lateral_acceleration': ('Vehicle.Acceleration.Lateral', 'float'),
}


@dataclass(frozen=True, slots=True)
class GraphWeights:
    version: int
    source_vehicles: tuple[str, ...]
    capture_hashes: tuple[str, ...]
    dbc_hashes: tuple[str, ...]
    feature_definitions: tuple[str, ...]
    validation: tuple[dict, ...]
    weights: tuple[tuple[str, float], ...]
    threshold: float = 0.0

    def check_target(self, vehicle: str | None, captures) -> None:
        if self.version != 1:
            raise ValueError('unsupported candidate graph artifact')
        if vehicle is None:
            raise ValueError('target vehicle identity is required for graph ranking')
        if vehicle.casefold() in {name.casefold() for name in self.source_vehicles}:
            raise ValueError('target vehicle appears in graph training sources')
        if {item.sha256 for item in captures} & set(self.capture_hashes):
            raise ValueError('target capture appears in graph training sources')
        if any(value < 0 or not np.isfinite(value) for _, value in self.weights):
            raise ValueError('graph weights must be finite and nonnegative')

    @classmethod
    def read(cls, path: Path) -> GraphWeights:
        data = json.loads(path.read_text())
        return cls(data['version'], tuple(data['source_vehicles']),
                   tuple(data['capture_hashes']), tuple(data['dbc_hashes']),
                   tuple(data['feature_definitions']), tuple(data['validation']),
                   tuple(tuple(row) for row in data['weights']), data['threshold'])

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + '\n')


def fit_graph_weights(sources: list[tuple], *,
                      target_vehicle: str) -> GraphWeights:
    """Fit nonnegative family weights using disjoint source validation trips."""
    from ...decoding.dbc import rank_dbc
    from .source_calibration import fit_source_calibration, fit_source_factor_validation
    if len(sources) < 2:
        raise ValueError('at least two source vehicles are required')
    normalized = []
    for source in sources:
        if len(source) == 3:
            name, trips, dbc = source
            fitting, validation = trips[:2], trips[2:]
        else:
            name, fitting, validation, dbc = source
        if len(fitting) != 2 or not validation or len({trip.sha256 for trip in (*fitting, *validation)}) != len(fitting) + len(validation):
            raise ValueError('graph fitting requires two fitting and independent validation trips')
        normalized.append((name, fitting, validation, dbc))
    if any(name.casefold() == target_vehicle.casefold() for name, _, _, _ in normalized):
        raise ValueError('target vehicle appears in graph training sources')
    validations = fit_source_factor_validation(normalized, target_vehicle=target_vehicle)
    calibrations = fit_source_calibration(
        [(name, fitting, dbc) for name, fitting, _, dbc in normalized],
        target_vehicle=target_vehicle)
    from ...decoding.dbc import load_dbc_fields
    from .source_calibration import _dbc_role
    from .unary_roles import _role_score, _sample_values, _series_metrics
    thresholds = {item.role: item.inclusion_threshold for item in calibrations}
    held_unary = []
    for name, _, validation_trips, dbc in normalized:
        assessed = retained = 0
        for field in load_dbc_fields(dbc):
            role = _dbc_role(field.label_hypotheses[0][0])
            if role not in thresholds:
                continue
            scores = []
            for trip in validation_trips:
                history = [_sample_values(trip, field, cap=256)]
                metrics = _series_metrics(history)
                if metrics is not None:
                    scores.append(_role_score(role, history, field, metrics)[0])
            if len(scores) != len(validation_trips):
                continue
            assessed += 1
            retained += all(score >= thresholds[role] for score in scores)
        held_unary.append({'vehicle': name, 'held_out_unary_assessed': assessed,
                           'held_out_unary_retained': int(retained),
                           'held_out_unary_recall': retained / assessed if assessed else None})
    effects: dict[str, list[float]] = {'physical': [], 'temporal': []}
    for item in validations:
        category = 'temporal' if item.name.startswith('graph:') else 'physical'
        effects[category].append(max(0.0, item.weight))
    unary_margin = [max(0., item.positive_mean - item.negative_mean) /
                    max(item.scale, 1e-6) for item in calibrations]
    held_coverage = [rank_dbc(trip, dbc)['coverage']
                     for _, _, validation, dbc in normalized for trip in validation]
    weights = (('unary', min(2.0, float(np.mean(unary_margin))) if unary_margin else 0.0),
               ('structural', .05 * float(np.mean(held_coverage))),
               ('temporal', min(.35, float(np.mean(effects['temporal']))) if effects['temporal'] else 0.0),
               ('physical', min(.35, float(np.mean(effects['physical']))) if effects['physical'] else 0.0))
    from hashlib import sha256
    threshold = (.5 * weights[0][1] * min(
        max(0., item.positive_mean - .45) for item in calibrations)
        if calibrations else 0.0)
    return GraphWeights(1, tuple(name for name, _, _, _ in normalized),
        tuple(capture.sha256 for _, fitting, validation, _ in normalized for capture in (*fitting, *validation)),
        tuple(sha256(path.read_bytes()).hexdigest() for _, _, _, path in normalized),
        ('unary:sum(max(score-.45,0))', 'structural:layout_count',
         'temporal:max_supported_effect_per_relationship',
         'physical:max_supported_effect_per_relationship'),
        tuple({'name': item.name, 'supporting_vehicles': item.supporting_vehicles,
               'weight': item.weight, 'third_trip_signatures': item.signatures}
              for item in validations) +
        tuple({'role': item.role, 'source_vehicles': item.source_vehicles,
               'positive_count': item.positive_count,
               'negative_count': item.negative_count,
               'leave_one_vehicle_out_recall': item.leave_one_vehicle_out_recall}
              for item in calibrations) + tuple(held_unary), weights, threshold)


@dataclass(frozen=True, slots=True)
class GraphNode:
    kind: str
    key: str
    provenance: str
    support: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphEvidence:
    family: str
    name: str
    members: tuple[str, ...]
    status: str
    effect: float
    provenance: str
    support: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphRanking:
    nodes: tuple[GraphNode, ...]
    evidence: tuple[GraphEvidence, ...]
    combinations: tuple[CandidateCombination, ...]
    layout_choices: tuple[dict, ...]


def _family(field: FieldCandidate) -> str:
    return ':'.join(map(str, (field.bus, field.arb_id, field.payload_length,
                              field.mux_start, field.mux_value)))


def selected_layout_fields(candidates: tuple[FieldCandidate, ...],
                           primary: tuple[FieldCandidate, ...],
                           memberships: dict[str, tuple[str, ...]],
                           choices: tuple[tuple[str, str], ...],
                           protocol: tuple[FieldCandidate, ...] = ()) -> tuple[FieldCandidate, ...]:
    """Materialize one decoded layout per family, with protocol bits reserved."""
    chosen = dict(choices)
    proposed = list(protocol)
    proposed += [field for field in primary if _family(field) not in chosen]
    proposed += [field for field in candidates if field.signed is not None and
                 chosen.get(_family(field)) in memberships.get(field_key(field), ())]
    result = []
    seen = set()
    owned: dict[str, set[int]] = {}
    for field in proposed:
        key, family = field_key(field), _family(field)
        if key in seen:
            continue
        bits = set(_bit_positions(field.start, field.width, field.endian))
        if bits & owned.get(family, set()):
            continue
        result.append(field)
        seen.add(key)
        owned.setdefault(family, set()).update(bits)
    return tuple(result)


def layout_memberships(fields: tuple[FieldCandidate, ...], tokenizations) -> dict[str, tuple[str, ...]]:
    """Map every candidate to all tokenized layouts that contain its exact bits."""
    result: dict[str, tuple[str, ...]] = {}
    by_family = {(item.trace.bus, item.trace.arb_id, item.trace.payload_length): item
                 for item in tokenizations}
    def matches(span, field):
        if span.byte_order != field.endian or len(span.physical_bits) != field.width:
            return False
        if span.byte_order == 'little':
            mapped = {(bit // 8) * 8 + 7 - bit % 8 for bit in span.physical_bits}
            return (max(mapped) - min(mapped) + 1 == len(mapped) and
                    min(mapped) == field.start)
        return min(span.physical_bits) == field.start
    for field in fields:
        token = by_family.get((field.bus, field.arb_id, field.payload_length))
        memberships = []
        if token:
            alternatives = token.alternatives or ()
            for index, alternative in enumerate(alternatives):
                for span in alternative.spans:
                    if matches(span, field):
                        memberships.append(f'{_family(field)}:layout:{index}')
            if not alternatives and any(matches(span, field) for span in token.spans):
                memberships.append(f'{_family(field)}:layout:0')
        result[field_key(field)] = tuple(memberships)
    return result


def _relationship_group(name: str) -> str:
    if 'speed' in name and ('brake' in name or 'throttle' in name):
        return 'longitudinal_control'
    if 'rpm' in name and ('throttle' in name or 'speed' in name):
        return 'powertrain'
    if 'steering' in name or 'yaw' in name or 'lateral' in name:
        return 'lateral_motion'
    return name.split(':')[-1]


def _compatible(combination, discovery, memberships, protocol_fields):
    layouts: dict[str, set[str]] = {}
    owned: dict[str, set[int]] = {}
    for index in combination.assignments:
        if index is None:
            continue
        field = discovery.hypotheses[index].candidate
        family = _family(field)
        choices = set(memberships.get(field_key(field), ()))
        if not choices:
            return None
        layouts[family] = layouts.get(family, choices) & choices
        if not layouts[family]:
            return None
        bits = set(_bit_positions(field.start, field.width, field.endian))
        if bits & owned.get(family, set()):
            return None
        owned.setdefault(family, set()).update(bits)
        if any(_family(p) == family and bits & _bit_positions(p.start, p.width, p.endian)
               for p in protocol_fields):
            return None
    return tuple(sorted((family, sorted(options)[0]) for family, options in layouts.items()))


def rerank(discovery: PhysicalDiscovery, memberships: dict[str, tuple[str, ...]],
           artifact: GraphWeights, *, max_combinations: int = 8) -> GraphRanking:
    """Apply source weights with one capped contribution per relationship family."""
    weights = dict(artifact.weights)
    nodes: dict[tuple[str, str], GraphNode] = {}
    evidence: list[GraphEvidence] = []
    scored = []
    for hypothesis in discovery.hypotheses:
        field = hypothesis.candidate
        family, key = _family(field), field_key(field)
        nodes[('family', family)] = GraphNode('family', family, 'target_capture')
        nodes[('field', key)] = GraphNode('field', key, 'tokenizer', hypothesis.support)
        nodes[('role', hypothesis.role)] = GraphNode('role', hypothesis.role,
            'vss:' + VSS_VERSION + ':' + ':'.join(VSS_ROLES[hypothesis.role]),
            hypothesis.support)
        for layout in memberships.get(key, ()):
            nodes[('layout', layout)] = GraphNode('layout', layout, 'tokenizer', hypothesis.support)
    for field in discovery.protocol_fields:
        key = field_key(field)
        nodes[('field', key)] = GraphNode('field', key, 'verified_protocol', field.reference_support)
        evidence.append(GraphEvidence('protocol', field.role, (key,), 'supported', 1.0,
                                      '|'.join(field.evidence), field.reference_support))
    duplicates: dict[tuple[str, int], list[str]] = {}
    for hypothesis in discovery.hypotheses:
        if hypothesis.equivalence_group >= 0:
            duplicates.setdefault((hypothesis.role, hypothesis.equivalence_group), []).append(
                field_key(hypothesis.candidate))
    for keys in duplicates.values():
        if len(set(keys)) > 1:
            evidence.append(GraphEvidence('duplicate', 'affine_trace', tuple(sorted(set(keys))),
                                          'supported', 1.0, 'capture_equivalence'))
    for combination in discovery.combinations:
        choices = _compatible(combination, discovery, memberships, discovery.protocol_fields)
        if choices is None:
            continue
        unary = sum(max(0.0, discovery.hypotheses[index].unary_score - .45)
                    for index in combination.assignments if index is not None)
        groups: dict[str, float] = {}
        for factor in combination.factors:
            family = _relationship_group(factor.name)
            evidence.append(GraphEvidence(family, factor.name,
                tuple(field_key(discovery.hypotheses[index].candidate) for index in factor.members),
                factor.status, factor.effect, 'target_candidate_factor',
                (factor.valid_samples,)))
            if factor.status == 'supported':
                category = 'temporal' if factor.name.startswith('graph:') else 'physical'
                contribution = weights.get(category, 0.0) * min(1.0, max(0.0, factor.effect))
                groups[family] = max(groups.get(family, 0.0), contribution)
        score = weights.get('unary', 0.0) * unary + sum(groups.values())
        score += weights.get('structural', 0.0) * len(choices)
        if score >= artifact.threshold:
            scored.append((score, combination, choices))
    scored.sort(key=lambda item: (-item[0], item[1].assignments.__repr__()))
    scored = scored[:max_combinations]
    if not scored:
        abstention = CandidateCombination((None,) * len(ROLES), 1.0, 0.0, (), ())
        return GraphRanking(tuple(nodes.values()), tuple(evidence),
                            (abstention,), ({'assignments': abstention.assignments,
                                             'layouts': ()},))
    raw = np.exp(np.array([row[0] for row in scored]) - scored[0][0])
    raw /= raw.sum()
    combinations = tuple(CandidateCombination(item.assignments, float(weight), float(score),
        item.relation_evidence, item.factors, item.runtime_weight)
        for (score, item, _), weight in zip(scored, raw, strict=True))
    layout_choices = tuple({'assignments': item.assignments, 'layouts': choices}
                           for _, item, choices in scored)
    return GraphRanking(tuple(nodes.values()), tuple(evidence), combinations, layout_choices)
