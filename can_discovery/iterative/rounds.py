"""Immutable round artifacts around the bounded physical-role search."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from ..manifests.common import artifact_hash, write_artifact
from ..manifests.iterative import freeze_manifest, verify_manifest

from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical import (
    ROLES,
    fit_physical_roles,
    fit_source_calibration,
    fit_source_factor_validation,
)
from can_discovery.relationships.physical.physics_factors import (
    FACTOR_ROLES,
    evaluate_factor,
)

SCHEMA = 'can-iterative-v1'
FACTOR_CAP = .35


def _captures(entry: dict, split: str) -> list:
    return [load_capture(Path(item['path'])) for item in entry['captures'][split]]


def fit_shared_model(manifest: dict, target_vehicle: str, *, round_number: int = 0,
                     previous: dict | None = None) -> dict:
    verify_manifest(manifest)
    sources = [entry for entry in manifest['vehicles'] if entry['vehicle'] != target_vehicle]
    unary_inputs = [(entry['vehicle'], _captures(entry, 'fitting'), Path(entry['dbc']['path']))
                    for entry in sources]
    factor_inputs = [(entry['vehicle'], _captures(entry, 'fitting'),
                      _captures(entry, 'development_validation'), Path(entry['dbc']['path']))
                     for entry in sources]
    unary = fit_source_calibration(unary_inputs, target_vehicle=target_vehicle)
    factor_diagnostics: list[dict] = []
    factors = fit_source_factor_validation(factor_inputs, target_vehicle=target_vehicle,
                                           diagnostics=factor_diagnostics)
    prior = previous['factor_weights'] if previous else {}
    weights = {factor.name: min(FACTOR_CAP, prior.get(factor.name, factor.weight))
               for factor in factors}
    return {'schema': SCHEMA, 'kind': 'shared_model', 'round': round_number,
            'manifest_hash': artifact_hash(manifest), 'target_vehicle': target_vehicle,
            'source_vehicles': [entry['vehicle'] for entry in sources],
            'source_hashes': {entry['vehicle']: {'dbc': entry['dbc']['sha256'],
                 'captures': [c['sha256'] for split in ('fitting', 'development_validation')
                              for c in entry['captures'][split]]} for entry in sources},
            'roles': list(ROLES), 'role_calibration': [asdict(item) for item in unary],
            'source_factors': [asdict(item) for item in factors],
            'factor_diagnostics': factor_diagnostics,
            'factor_gate': {name: {
                'supporting_vehicles': sorted(item['vehicle'] for item in factor_diagnostics
                                              if item['factor'] == name and item['status'] == 'supported'),
                'status': 'eligible' if name in weights else 'rejected',
                'reason': None if name in weights else 'two_vehicle_compatible_gate_failed',
            } for name in FACTOR_ROLES},
            'factor_weights': weights, 'parent_hash': artifact_hash(previous) if previous else None,
            'promotion_records': previous.get('promotion_records', []) if previous else []}


def _field(field) -> dict:
    keys = ('bus', 'arb_id', 'payload_length', 'start', 'width', 'endian', 'signed',
            'mux_start', 'mux_value', 'role')
    return {key: getattr(field, key) for key in keys}


def infer_vehicle(entry: dict, shared: dict, *, track: str = 'source_only',
                  anchor_roles: tuple[str, ...] = ()) -> dict:
    if track not in ('source_only', 'target_anchor_assisted'):
        raise ValueError(track)
    if track == 'source_only' and anchor_roles:
        raise ValueError('source-only track cannot expose target anchors')
    if entry['vehicle'] != shared['target_vehicle']:
        raise ValueError('shared model target mismatch')
    from can_discovery.decoding.dbc import load_dbc_fields
    from can_discovery.decoding.signals import propose_fields
    from can_discovery.relationships.physical import (
        SourceFactorValidation,
        SourceRoleCalibration,
    )
    from can_discovery.relationships.physical.source_calibration import _dbc_role
    fitting = _captures(entry, 'fitting')
    candidates = propose_fields(fitting, max_ids=64)
    anchors = []
    if track == 'target_anchor_assisted':
        if not anchor_roles:
            raise ValueError('assisted track requires declared anchor roles')
        anchors = [field for field in load_dbc_fields(Path(entry['dbc']['path']))
                   if _dbc_role(field.label_hypotheses[0][0]) in anchor_roles]
        candidates = tuple(candidates) + tuple(replace(field, role='unknown') for field in anchors)
    factors = tuple(SourceFactorValidation(**{**item,
        'weight': shared['factor_weights'][item['name']]})
        for item in shared['source_factors'] if item['name'] in shared['factor_weights'])
    discovery = fit_physical_roles(fitting, candidates=candidates,
        source_calibration=tuple(SourceRoleCalibration(**item) for item in shared['role_calibration']),
        validated_factors=factors, max_role_candidates=64, max_family_candidates=8,
        max_combinations=8)
    proposals = []
    for index, combination in enumerate(discovery.combinations):
        assignments = {}
        for role, hypothesis_index in zip(ROLES, combination.assignments, strict=True):
            hypothesis = discovery.hypotheses[hypothesis_index] if hypothesis_index is not None else None
            assignments[role] = {'field': _field(hypothesis.candidate),
                                 'unary_score': hypothesis.unary_score,
                                 'support': hypothesis.support,
                                 'equivalence_group': hypothesis.equivalence_group,
                                 'alternatives': [role, 'unknown'],
                                 'evidence': hypothesis.evidence} if hypothesis else None
        proposals.append({'rank': index + 1, 'assignments': assignments,
                          'score': combination.score, 'weight': combination.weight,
                          'score_components': dict(combination.relation_evidence),
                          'factor_evidence': [asdict(item) for item in combination.factors],
                          'status': 'provisional', 'parent_hypotheses': [],
                          'abstention_reason': 'null_assignment' if all(v is None for v in assignments.values()) else None})
    return {'schema': SCHEMA, 'kind': 'vehicle_proposals', 'round': shared['round'],
            'shared_hash': artifact_hash(shared), 'vehicle': entry['vehicle'],
            'track': track, 'declared_anchor_roles': list(anchor_roles),
            'fitting_hashes': [c['sha256'] for c in entry['captures']['fitting']],
            'validation_hashes': [c['sha256'] for c in entry['captures']['development_validation']],
            'proposals': proposals, 'candidate_count': len(candidates),
            'rejected': discovery.rejected}


def validate_proposals(entry: dict, proposals: dict, shared: dict) -> dict:
    """Validate selected groups with independent trips and factor controls."""
    if proposals['shared_hash'] != artifact_hash(shared):
        raise ValueError('proposal/shared model hash mismatch')
    from can_discovery.decoding.signals import FieldCandidate
    validation = _captures(entry, 'development_validation')
    records = []
    for proposal in proposals['proposals']:
        for source in shared['source_factors']:
            name = source['name']
            roles = FACTOR_ROLES.get(name)
            if not roles:
                continue  # Graph and model factors remain diagnostic here.
            members = [proposal['assignments'][role] for role in roles]
            if any(member is None for member in members):
                continue
            fields = tuple(FieldCandidate(arb_id=member['field']['arb_id'], start=member['field']['start'],
                width=member['field']['width'], endian=member['field']['endian'],
                signed=member['field']['signed'], role='unknown', limit=0., confidence=0.,
                reference_support=(), label_hypotheses=(), evidence=(),
                payload_length=member['field']['payload_length'],
                mux_start=member['field']['mux_start'], mux_value=member['field']['mux_value'],
                bus=member['field']['bus']) for member in members)
            results = [evaluate_factor([capture], fields, name) for capture in validation]
            passed = all(item.status == 'supported' and
                         item.effect > item.control_effect and item.effect > 0
                         for item in results)
            records.append({'factor': name, 'vehicle': entry['vehicle'],
                            'proposal_rank': proposal['rank'], 'passed': passed,
                            'validation_hashes': proposals['validation_hashes'],
                            'ancestry': [proposals['shared_hash']],
                            'support': [asdict(item) for item in results],
                            'reason': 'replicated_controls_passed' if passed else 'validation_or_control_failed'})
    return {'schema': SCHEMA, 'kind': 'validation', 'round': shared['round'],
            'proposal_hash': artifact_hash(proposals), 'records': records}


def promote_round(shared: dict, validations: list[dict]) -> dict:
    """Apply bounded soft updates only to source validated, independent evidence."""
    updated = dict(shared['factor_weights'])
    ledger = []
    source = {item['name']: item for item in shared['source_factors']}
    for validation in validations:
        for item in validation['records']:
            name = item['factor']
            reason = None
            if name not in source or len(source[name]['supporting_vehicles']) < 2:
                reason = 'source_gate_unavailable'
            elif not item['passed']:
                reason = item['reason']
            elif artifact_hash(shared) in item['ancestry'] and item['vehicle'] in shared['source_vehicles']:
                reason = 'circular_source_ancestry'
            elif not item['validation_hashes'] or set(item['validation_hashes']) & {
                    h for v in shared['source_hashes'].values() for h in v['captures']}:
                reason = 'nonindependent_validation'
            old = updated.get(name, 0.)
            new = old if reason else min(FACTOR_CAP, old + .025)
            updated[name] = new
            ledger.append({'factor': name, 'vehicle': item['vehicle'], 'old_weight': old,
                           'new_weight': new, 'passed': reason is None,
                           'reason': reason or 'bounded_soft_update',
                           'validation_hashes': item['validation_hashes'],
                           'proposal_rank': item['proposal_rank']})
    result = dict(shared)
    result.update(round=shared['round'] + 1, parent_hash=artifact_hash(shared),
                  factor_weights=updated,
                  promotion_records=shared['promotion_records'] + ledger)
    return result
