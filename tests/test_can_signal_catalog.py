"""Regression checks for engineering catalog role compatibility and evidence gates."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from can_discovery.decoding.signals import FieldCandidate
from can_discovery.engineering import catalog
from can_discovery.preprocessing.capture import Capture, Frame


def _field(start: int) -> FieldCandidate:
    return FieldCandidate('100', start, 8, 'little', False, 'unknown', 0., 1.,
                          (100, 100), (('unknown', 1.),), ('synthetic',), 8)


def test_joint_match_does_not_assign_one_field_to_two_roles():
    shared = _field(0)
    other = _field(8)
    options = {role: {'anchors': [], 'candidates': []} for role in catalog.ROLES}
    options['accelerator_pedal']['candidates'] = [
        {'rank': 1, 'score': .9, 'field': asdict(shared), 'status': 'provisional'}]
    options['throttle_plate']['candidates'] = [
        {'rank': 1, 'score': .8, 'field': asdict(shared), 'status': 'provisional'},
        {'rank': 2, 'score': .7, 'field': asdict(other), 'status': 'provisional'}]
    best = catalog._joint_matches(options)[0]['assignments']
    assert best['accelerator_pedal'] == 1
    assert best['throttle_plate'] == 2


def test_catalog_runner_never_opens_declared_final_route(tmp_path, monkeypatch):
    from scripts import mine_can_signal_catalog as runner

    dbc = tmp_path / 'vehicle.dbc'
    dbc.write_text('VERSION ""\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'vehicle': 'VEHICLE',
        'fitting': ['fit_1.csv', 'fit_2.csv'], 'validation': ['dev.csv'],
        'final': 'sealed.csv', 'dbc': str(dbc)}]))
    opened = []

    def load(path: Path):
        opened.append(str(path))
        return object()

    def mine(vehicle, fitting, validation, path, **kwargs):
        assert len(fitting) == 2 and len(validation) == 1
        return {'vehicle': vehicle, 'candidate_count': 0,
                'catalog': {role: {'status': 'unobserved'} for role in catalog.ROLES},
                'recovery_gate': {'ready': False},
                'leave_one_role_out': {}}

    monkeypatch.setattr(runner, 'load_capture', load)
    monkeypatch.setattr(runner, 'mine_vehicle', mine)
    runner.run(manifest, tmp_path / 'output')
    assert opened == ['fit_1.csv', 'fit_2.csv', 'dev.csv']


def test_value_equivalence_requires_independent_route_agreement():
    left, right = _field(0), _field(8)

    def capture(name: str, gain: int) -> Capture:
        frames = tuple(Frame(i*.05, i, '100', bytes((i % 100, gain*(i % 100)+3,
                                                    0, 0, 0, 0, 0, 0)))
                       for i in range(220))
        return Capture(name, name, frames, 0., 219*.05, 0, 0, {8: 220}, {16: 220})

    fitting = [capture('fit1', 2), capture('fit2', 2)]
    assert catalog._value_equivalent(left, right, fitting)
    assert not catalog._value_equivalent(left, right, [*fitting, capture('dev', 1)])
