from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from can_discovery.decoding.signals import FieldCandidate, FieldRegistry
from can_discovery.preprocessing.capture import Capture, Frame
from can_discovery.relationships.physical.role_types import (
    PhysicalDiscovery,
    RoleHypothesis,
)
from can_discovery.relationships.physical.signal_graph import (
    _edge,
    _field_layouts,
    build_signal_graph,
)


def _field(arb_id: str) -> FieldCandidate:
    return FieldCandidate(arb_id, 0, 16, 'little', False, 'unknown', 0., 1., (),
                          (('unknown', 1.),), (), 2)


def _captures(*, independent: bool = False):
    captures = []
    for seed in (11, 23):
        rng = np.random.default_rng(seed)
        one = np.cumsum(rng.integers(-3, 4, 800)) + 500
        two = (np.cumsum(rng.integers(-3, 4, 800)) + 700 if independent
               else one)
        frames = []
        for i in range(800):
            for j, (arb_id, value) in enumerate((
                ('100', one[i]), ('200', one[max(0, i-2)]), ('300', two[i]))):
                frames.append(Frame(i*.02+j*.001, 3*i+j, arb_id,
                                    int(value).to_bytes(2, 'little')))
        captures.append(Capture(str(seed), str(seed), tuple(frames), 0., 16.,
                                0, 0, {}, {}))
    return captures


def test_delayed_copy_has_upstream_observed_root():
    captures = _captures()
    fields = tuple(_field(arb_id) for arb_id in ('100', '200'))
    edge = _edge(captures, *fields, 0, 1)
    assert edge.status == 'supported'
    assert edge.relation == 'exact_copy'
    assert (edge.source, edge.target) == (0, 1)
    assert edge.lag_seconds > 0
    assert len(edge.support) == 2
    discovery = PhysicalDiscovery(tuple(RoleHypothesis(field, 'speed', .9-i*.1,
        (), ()) for i, field in enumerate(fields)), (), (), ())
    graph = build_signal_graph(captures, FieldRegistry(fields, fields), discovery)
    assert graph['version'] == 2
    assert 'conditional_layouts' not in graph
    group = next(item for item in graph['role_groups'] if item['role'] == 'speed')
    assert group['members'] == [0, 1]
    assert group['observed_roots'] == [0]
    assert group['origin'] == 'unknown'
    assert 'rpm' in graph['null_roles']


def test_synchronous_equivalents_keep_both_roots():
    captures = _captures()
    fields = tuple(_field(arb_id) for arb_id in ('100', '300'))
    edge = _edge(captures, *fields, 0, 1)
    assert edge.relation == 'synchronous_equivalent'
    discovery = PhysicalDiscovery(tuple(RoleHypothesis(field, 'speed', .9,
        (), ()) for field in fields), (), (), ())
    graph = build_signal_graph(captures, FieldRegistry(fields, fields), discovery)
    group = next(item for item in graph['role_groups'] if item['role'] == 'speed')
    assert group['observed_roots'] == [0, 1]
