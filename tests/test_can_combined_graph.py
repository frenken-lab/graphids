"""The modular graph preserves missing evidence and freezes assessment choices."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from can_discovery.relationships.physical import combined_graph as graph
from can_discovery.relationships.shared.evidence import FactorEvidence


def _capture(digest):
    return SimpleNamespace(sha256=digest)


def test_missing_sensor_abstains_without_failing_other_modules(monkeypatch):
    calls = []
    def evaluate(captures, fields, name, forced_lag=None):
        calls.append((name, forced_lag))
        return FactorEvidence(name, graph.FACTOR_ROLES[name], 'supported', .2,
                              lag_bins=2)
    monkeypatch.setattr(graph, 'evaluate_factor', evaluate)
    fitted = graph.evaluate_combined_graph([_capture('a'), _capture('b')],
                                           {'speed': object(), 'brake': object()}, powertrain='ICE')
    assert fitted['modules']['braking']['status'] == 'supported'
    assert fitted['modules']['chassis']['status'] == 'unavailable'
    assert fitted['fit_choices']['esc_torque_feedback']['reason'].startswith('missing_role:')
    assert calls == [('speed_brake', None)]


def test_development_uses_frozen_lag_and_cannot_rescue_missing_fit_factor(monkeypatch):
    calls = []
    def evaluate(captures, fields, name, forced_lag=None):
        calls.append((name, forced_lag))
        return FactorEvidence(name, graph.FACTOR_ROLES[name], 'supported', .2,
                              lag_bins=3)
    monkeypatch.setattr(graph, 'evaluate_factor', evaluate)
    fields = {'speed': object(), 'brake': object(), 'throttle': object()}
    fitted = graph.evaluate_combined_graph([_capture('fit')], fields, powertrain='ICE')
    calls.clear()
    fields['rpm'] = object()
    development = graph.evaluate_combined_graph([_capture('dev')], fields,
                                                 frozen=fitted['fit_choices'], powertrain='ICE')
    assert ('speed_brake', 3) in calls
    assert ('speed_throttle', 3) in calls
    assert all(name != 'rpm_throttle' for name, _ in calls)
    assert development['fit_choices']['rpm_throttle']['reason'] == 'unavailable_at_fit'


def test_runner_ignores_target_dbc_and_rejects_assessment_overlap(monkeypatch):
    from can_discovery.relationships.physical.role_types import ROLES
    from scripts import benchmark_can_combined_graph as runner

    calls = []
    def load(entry, role):
        calls.append(role)
        return [_capture('fit-a'), _capture('fit-b')] if role == 'fitting' else [_capture('fit-a')]
    monkeypatch.setattr(runner, '_load', load)
    from can_discovery.decoding.signals import FieldCandidate
    field = FieldCandidate('100', 0, 8, 'little', False, 'unknown', 0., 0., (), (), ())
    hypothesis = SimpleNamespace(candidate=field)
    combination = SimpleNamespace(assignments=(0,) + (None,) * (len(ROLES)-1),
                                  weight=1., score=1.)
    monkeypatch.setattr(runner, 'fit_physical_roles',
                        lambda captures, max_ids: SimpleNamespace(
                            combinations=[combination], hypotheses=[hypothesis]))
    monkeypatch.setattr(runner, 'evaluate_combined_graph',
                        lambda captures, fields, **kwargs: {
                            'capture_hashes': [c.sha256 for c in captures], 'fit_choices': {}})
    with pytest.raises(ValueError, match='overlap'):
        runner.run_entry({'vehicle': 'Civic', 'dbc': '/does/not/exist'}, sealed=False)
    assert calls == ['fitting', 'development']
