"""Experimental modular vehicle graph over fit-only physical role candidates.

The graph records physical hypotheses, not a PicaCAN SECG replacement. A factor
is available only when every required role has an independently selected field.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from ...decoding.signals import FieldCandidate
from ...preprocessing.capture import Capture
from ..shared.evidence import FactorEvidence
from .physics_factors import FACTOR_ROLES, evaluate_factor
from .vehicle_models import MODEL_ROLES, evaluate_vehicle_model

Module = Literal['powertrain', 'braking', 'chassis', 'interface']


@dataclass(frozen=True, slots=True)
class GraphFactor:
    name: str
    module: Module
    roles: tuple[str, ...]
    kind: Literal['lagged_response', 'algebraic', 'equivalence', 'conditional']
    regime: str
    evaluator: str | None = None
    powertrains: tuple[str, ...] = ()


# A latent node is a physical quantity for which this bus may have no CAN field.
NODES = {
    'powertrain': ('throttle', 'rpm', 'gear', 'propulsion_torque'),
    'braking': ('brake', 'wheel_brake_torque'),
    'chassis': ('steering', 'speed', 'wheel_speed', 'yaw',
                'lateral_acceleration', 'acceleration'),
    'interface': ('propulsion_torque', 'wheel_brake_torque', 'speed',
                  'acceleration', 'esc_torque_request', 'esc_brake_request'),
}

FACTORS = (
    GraphFactor('speed_rpm_gear', 'powertrain', ('speed', 'rpm', 'gear'),
                'conditional', 'in_gear_low_slip', 'factor', ('ICE',)),
    GraphFactor('rpm_throttle', 'powertrain', ('rpm', 'throttle'),
                'lagged_response', 'ordinary_driving', 'factor', ('ICE',)),
    GraphFactor('speed_brake', 'braking', ('speed', 'brake'),
                'lagged_response', 'ordinary_braking_no_abs', 'factor'),
    GraphFactor('wheel_speed', 'chassis', ('speed', 'wheel_speed'),
                'equivalence', 'straight_low_slip', 'factor'),
    GraphFactor('model:kinematic_yaw', 'chassis', ('speed', 'steering', 'yaw'),
                'algebraic', 'moderate_speed_low_slip', 'model'),
    GraphFactor('model:kinematic_lateral', 'chassis',
                ('speed', 'yaw', 'lateral_acceleration'),
                'algebraic', 'moderate_speed_low_slip', 'model'),
    GraphFactor('steering_yaw', 'chassis', ('speed', 'steering', 'yaw'),
                'lagged_response', 'moderate_speed_low_slip', 'factor'),
    GraphFactor('steering_lateral', 'chassis',
                ('speed', 'steering', 'lateral_acceleration'),
                'lagged_response', 'moderate_speed_low_slip', 'factor'),
    GraphFactor('speed_acceleration', 'interface', ('speed', 'acceleration'),
                'algebraic', 'ordinary_driving', 'factor'),
    GraphFactor('speed_throttle', 'interface', ('speed', 'throttle'),
                'lagged_response', 'ordinary_driving', 'factor'),
    GraphFactor('propulsion_to_motion', 'interface',
                ('propulsion_torque', 'acceleration'), 'lagged_response',
                'ordinary_driving'),
    GraphFactor('brake_torque_to_motion', 'interface',
                ('wheel_brake_torque', 'acceleration'), 'lagged_response',
                'ordinary_braking_no_abs'),
    GraphFactor('esc_torque_feedback', 'interface',
                ('yaw', 'esc_torque_request', 'propulsion_torque'),
                'lagged_response', 'esc_intervention'),
    GraphFactor('esc_brake_feedback', 'interface',
                ('yaw', 'esc_brake_request', 'wheel_brake_torque'),
                'lagged_response', 'esc_intervention'),
)


@dataclass(frozen=True, slots=True)
class GraphLink:
    source: str
    target: str
    module: Module
    kind: Literal['actuation', 'response', 'estimate', 'feedback', 'condition']
    regime: str


# Physical wiring hypotheses. Support comes from FACTORS, never from mere
# presence of a link in this schema.
LINKS = (
    GraphLink('throttle', 'propulsion_torque', 'powertrain', 'actuation', 'ordinary_driving'),
    GraphLink('rpm', 'propulsion_torque', 'powertrain', 'condition', 'ICE_in_gear'),
    GraphLink('gear', 'propulsion_torque', 'powertrain', 'condition', 'ICE_in_gear'),
    GraphLink('brake', 'wheel_brake_torque', 'braking', 'actuation', 'ordinary_braking'),
    GraphLink('propulsion_torque', 'acceleration', 'interface', 'response', 'ordinary_driving'),
    GraphLink('wheel_brake_torque', 'acceleration', 'interface', 'response', 'ordinary_braking'),
    GraphLink('acceleration', 'speed', 'interface', 'response', 'ordinary_driving'),
    GraphLink('wheel_speed', 'speed', 'chassis', 'estimate', 'straight_low_slip'),
    GraphLink('steering', 'yaw', 'chassis', 'response', 'moderate_speed_low_slip'),
    GraphLink('speed', 'yaw', 'chassis', 'condition', 'moderate_speed_low_slip'),
    GraphLink('yaw', 'lateral_acceleration', 'chassis', 'response', 'moderate_speed_low_slip'),
    GraphLink('speed', 'lateral_acceleration', 'chassis', 'condition', 'moderate_speed_low_slip'),
    GraphLink('yaw', 'esc_torque_request', 'interface', 'feedback', 'esc_intervention'),
    GraphLink('yaw', 'esc_brake_request', 'interface', 'feedback', 'esc_intervention'),
    GraphLink('esc_torque_request', 'propulsion_torque', 'interface', 'feedback', 'esc_intervention'),
    GraphLink('esc_brake_request', 'wheel_brake_torque', 'interface', 'feedback', 'esc_intervention'),
)


def validate_graph() -> None:
    for link in LINKS:
        if link.source not in {node for group in NODES.values() for node in group} or \
                link.target not in {node for group in NODES.values() for node in group}:
            raise ValueError(f'invalid combined graph link: {link}')
    names = set()
    roles = {role for group in NODES.values() for role in group}
    for factor in FACTORS:
        if factor.name in names or not set(factor.roles) <= roles:
            raise ValueError(f'invalid combined graph factor: {factor.name}')
        names.add(factor.name)
        if factor.evaluator == 'factor' and FACTOR_ROLES[factor.name] != factor.roles:
            raise ValueError(f'factor role mismatch: {factor.name}')
        if factor.evaluator == 'model' and MODEL_ROLES[factor.name] != factor.roles:
            raise ValueError(f'model role mismatch: {factor.name}')


def _evaluate(spec: GraphFactor, captures: list[Capture], fields: dict[str, FieldCandidate],
              *, frozen_lag: int | None = None) -> FactorEvidence:
    missing = [role for role in spec.roles if role not in fields]
    if missing:
        return FactorEvidence(spec.name, spec.roles, 'unavailable', reason='missing_role:' + ','.join(missing))
    if spec.evaluator is None:
        return FactorEvidence(spec.name, spec.roles, 'unavailable', reason='no_validated_evaluator')
    selected = tuple(fields[role] for role in spec.roles)
    if spec.evaluator == 'factor':
        return evaluate_factor(captures, selected, spec.name, forced_lag=frozen_lag)
    return evaluate_vehicle_model(captures, selected, spec.name, forced_lag=frozen_lag)


def evaluate_combined_graph(captures: list[Capture], fields: dict[str, FieldCandidate],
                            *, frozen: dict[str, dict] | None = None,
                            powertrain: str = 'unknown') -> dict:
    """Evaluate per module; an assessment uses only fitted available lags.

    `fields` and `frozen` must be outputs of a prior fit. The target DBC is
    deliberately absent from this interface.
    """
    validate_graph()
    if not captures:
        raise ValueError('at least one capture is required')
    if len({capture.sha256 for capture in captures}) != len(captures):
        raise ValueError('captures must be distinct')
    unknown = set(fields) - {role for group in NODES.values() for role in group}
    if unknown:
        raise ValueError(f'unknown graph roles: {sorted(unknown)}')
    if powertrain not in ('ICE', 'EV', 'hybrid', 'unknown'):
        raise ValueError('unsupported powertrain label')
    by_module = {module: [] for module in NODES}
    for spec in FACTORS:
        fitted = frozen.get(spec.name) if frozen is not None else None
        if spec.powertrains and powertrain not in spec.powertrains:
            evidence = FactorEvidence(spec.name, spec.roles, 'unavailable',
                                      reason='powertrain_not_verified_or_inapplicable')
        elif frozen is not None and (fitted is None or fitted['status'] == 'unavailable'):
            evidence = FactorEvidence(spec.name, spec.roles, 'unavailable',
                                      reason='unavailable_at_fit')
        else:
            lag = fitted['lag_bins'] if fitted is not None else None
            evidence = _evaluate(spec, captures, fields, frozen_lag=lag)
        by_module[spec.module].append({**asdict(spec), 'evidence': asdict(evidence)})
    summary = {}
    for module, factors in by_module.items():
        counts = {status: sum(item['evidence']['status'] == status for item in factors)
                  for status in ('supported', 'failed', 'unavailable')}
        summary[module] = {'factors': factors, 'counts': counts,
                           'status': ('supported' if counts['supported'] == len(factors) else
                                      'partial' if counts['supported'] else
                                      'failed' if counts['failed'] else 'unavailable')}
    return {'version': 1, 'nodes': NODES,
            'links': [asdict(link) for link in LINKS], 'modules': summary,
            'powertrain': powertrain,
            'regime_assessment': 'whole_route_unfiltered; conditional factors are provisional',
            'capture_hashes': [capture.sha256 for capture in captures],
            'fit_choices': {item['name']: item['evidence']
                            for module in summary.values() for item in module['factors']},
            'promotion_status': 'experimental_unpromoted'}
