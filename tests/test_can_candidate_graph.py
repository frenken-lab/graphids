import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from can_discovery.decoding.signals import FieldCandidate
from can_discovery.relationships.physical import (
    CandidateCombination,
    PhysicalDiscovery,
    RoleHypothesis,
)
from can_discovery.relationships.physical.candidate_graph import (
    GraphWeights,
    field_key,
    layout_memberships,
    rerank,
    selected_layout_fields,
)
from can_discovery.relationships.shared.evidence import FactorEvidence


def _field(start, width=8, role='unknown'):
    return FieldCandidate('123', start, width, 'little', False, role, 0., 1.,
                          (100, 100), (('unknown', 1.),), ('test',), 2)


def _weights():
    return GraphWeights(1, ('source',), ('source-hash',), ('dbc-hash',),
                        ('unary', 'temporal'), (),
                        (('unary', 1.), ('structural', 0.),
                         ('temporal', 1.), ('physical', 1.)))


def _discovery(factors=(), protocol=()):
    fields = (_field(0), _field(8))
    hypotheses = tuple(RoleHypothesis(field, role, .8, (100, 100), ('test',))
                       for field, role in zip(fields, ('speed', 'brake')))
    assignment = (0, None, None, None, None, 1, None, None, None)
    combination = CandidateCombination(assignment, 1., 1., (), factors)
    return PhysicalDiscovery(hypotheses, (combination,), protocol, ())


def test_layout_exclusivity_and_protocol_conflict():
    discovery = _discovery()
    keys = [field_key(item.candidate) for item in discovery.hypotheses]
    memberships = {keys[0]: ('family:layout:0',), keys[1]: ('family:layout:1',)}
    assert all(index is None for index in
               rerank(discovery, memberships, _weights()).combinations[0].assignments)
    memberships[keys[1]] = ('family:layout:0',)
    assert rerank(discovery, memberships, _weights()).combinations
    assert all(index is None for index in rerank(
        replace(discovery, protocol_fields=(_field(0, role='counter'),)),
        memberships, _weights()).combinations[0].assignments)


def test_artifact_leakage_and_serialization(tmp_path):
    artifact = _weights()
    path = tmp_path / 'weights.json'
    artifact.write(path)
    assert GraphWeights.read(path) == artifact
    with pytest.raises(ValueError, match='target vehicle'):
        artifact.check_target('SOURCE', ())
    with pytest.raises(ValueError, match='target capture'):
        artifact.check_target('target', (SimpleNamespace(sha256='source-hash'),))
    artifact.check_target('target', (SimpleNamespace(sha256='other'),))


def test_membership_uses_exact_alternative_layout():
    field = _field(0)
    span = SimpleNamespace(physical_bits=tuple(range(7, -1, -1)), byte_order='little')
    token = SimpleNamespace(trace=SimpleNamespace(bus=None, arb_id='123', payload_length=2),
        alternatives=(SimpleNamespace(spans=(span,)),), spans=())
    result = layout_memberships((field,), (token,))
    assert result[field_key(field)] == ('None:123:2:None:None:layout:0',)
