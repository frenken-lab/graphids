from __future__ import annotations

import math

import numpy as np

from can_discovery.decoding.signals import FieldCandidate
from can_discovery.preprocessing.capture import Capture, Frame
from can_discovery.relationships.physical import combination_ranker as roles


def _capture(seed):
    frames = tuple(Frame(i * .1, i, '123',
                         bytes((i * (j + 3) + seed * j + i * i // (j + 2)) % 220
                                for j in range(8))) for i in range(160))
    return Capture(str(seed), str(seed), frames, 0, 15.9, 0, 0, {8: 160}, {16: 160})


def _field(start, signed=False):
    return FieldCandidate('123', start, 8, 'little', signed, 'unknown', 0., .8,
                          (160, 160), (('unknown', 1.),), ('candidate_lattice',), 8)


def test_threshold_pool_reports_every_overflow_and_groups_aliases(monkeypatch):
    fields = tuple(_field(8 * i) for i in range(8)) + (_field(0),)
    monkeypatch.setattr(roles, 'propose_fields', lambda *args, **kwargs: fields)
    monkeypatch.setattr(roles, '_role_score', lambda role, history, field, metrics:
                        (.5 - field.start * .001, ('controlled',)) if role == 'speed'
                        else (-math.inf, ()))
    found = roles.fit_physical_roles([_capture(1), _capture(2)], max_role_candidates=4,
                                     max_family_candidates=3)
    chosen = [item for item in found.hypotheses if item.role == 'speed']
    overflow = [item for item in found.rejected if item['reason'] == 'candidate_budget_overflow']
    assert len(chosen) == 3
    assert len(overflow) == 5  # The signed alias does not consume a slot.
    assert all(item['score'] >= item['threshold'] for item in overflow)
    assert any(item['reason'] == 'pool_threshold' and item['threshold'] == .30
               for item in found.rejected)


def test_completed_graph_can_rescue_weaker_unary_candidate(monkeypatch):
    speed_a, speed_b, acceleration = _field(0), _field(8), _field(16)
    monkeypatch.setattr(roles, 'propose_fields', lambda *args, **kwargs:
                        (speed_a, speed_b, acceleration))
    def score(role, history, field, metrics):
        if role == 'speed' and field.start in (0, 8):
            return (.8 if field.start == 0 else .65), ()
        if role == 'acceleration' and field.start == 16:
            return .8, ()
        return -math.inf, ()
    monkeypatch.setattr(roles, '_role_score', score)
    from can_discovery.relationships.physical.physics_factors import (
        FactorEvidence,
    )
    monkeypatch.setattr(roles, 'evaluate_factor', lambda captures, fields, name, indices=None:
                        FactorEvidence(name, roles.FACTOR_ROLES[name], 'supported',
                                       1.0 if name == 'speed_acceleration' and fields[0].start == 8 else 0.0))
    found = roles.fit_physical_roles([_capture(1), _capture(2)], beam_width=1,
                                     max_combinations=3)
    top = found.combinations[0]
    speed = top.assignments[roles.ROLES.index('speed')]
    assert speed is not None and found.hypotheses[speed].candidate.start == 8


def test_layout_membership_is_enforced_during_combination_search(monkeypatch):
    from can_discovery.relationships.physical.candidate_graph import field_key
    speed, brake = _field(0), _field(8)
    monkeypatch.setattr(roles, '_role_score', lambda role, history, field, metrics:
        (.8, ()) if (role, field.start) in (('speed', 0), ('brake', 8))
        else (-math.inf, ()))
    memberships = {field_key(speed): ('layout:0',),
                   field_key(brake): ('layout:1',)}
    found = roles.fit_physical_roles([_capture(1), _capture(2)],
        candidates=(speed, brake), layout_memberships=memberships)
    assert all(sum(index is not None for index in combo.assignments) <= 1
               for combo in found.combinations)


def test_graph_edges_require_source_validation(monkeypatch):
    from can_discovery.relationships.physical.observational_edges import GraphEdgeEvidence
    speed, throttle = _field(0), _field(16)
    monkeypatch.setattr(roles,'propose_fields',lambda *args,**kwargs:(speed,throttle))
    monkeypatch.setattr(roles,'_role_score',lambda role,history,field,metrics:
                        (.8,()) if (role,field.start) in {('speed',0),('throttle',16)}
                        else (-math.inf,()))
    calls = []
    def edge(*args,**kwargs):
        calls.append(args[3])
        return GraphEdgeEvidence('throttle','speed','supported',.3,0.,.08,(100,100))
    monkeypatch.setattr(roles,'evaluate_graph_edge',edge)
    monkeypatch.setattr(roles,'duplicate_signal_groups',lambda hypotheses,captures,groups:
                        (groups,set(range(len(hypotheses)))))
    captures = [_capture(1),_capture(2)]
    roles.fit_physical_roles(captures)
    assert not calls
    validation = roles.SourceFactorValidation('graph:throttle_speed',('a','b'),.1,())
    found = roles.fit_physical_roles(captures,validated_factors=(validation,))
    assert 'throttle_speed' in calls
    assert any(f.name == 'graph:throttle_speed' and f.status == 'supported'
               for f in found.factor_evidence)


def test_vehicle_model_requires_source_validation(monkeypatch):
    from can_discovery.relationships.physical.physics_factors import FactorEvidence
    speed, acceleration = _field(0), _field(16)
    monkeypatch.setattr(roles,'propose_fields',lambda *args,**kwargs:(speed,acceleration))
    monkeypatch.setattr(roles,'_role_score',lambda role,history,field,metrics:
                        (.8,()) if (role,field.start) in {('speed',0),('acceleration',16)}
                        else (-math.inf,()))
    calls = []
    def model(captures,fields,name):
        calls.append(name)
        return FactorEvidence(name,roles.MODEL_ROLES[name],'supported',.2)
    monkeypatch.setattr(roles,'evaluate_vehicle_model',model)
    captures = [_capture(1),_capture(2)]
    roles.fit_physical_roles(captures)
    assert not calls
    validation = roles.SourceFactorValidation('model:double_integrator',('a','b'),.1,())
    found = roles.fit_physical_roles(captures,validated_factors=(validation,))
    assert 'model:double_integrator' in calls
    assert any(f.name == 'model:double_integrator' for f in found.factor_evidence)
