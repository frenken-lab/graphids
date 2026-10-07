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
