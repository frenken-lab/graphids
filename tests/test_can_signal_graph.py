from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from can_discovery.decoding.signals import FieldCandidate, FieldRegistry
from can_discovery.preprocessing.capture import Capture, Frame
from can_discovery.preprocessing.tokenization import MessageTrace, tokenize
from can_discovery.relationships.physical.role_types import (
    PhysicalDiscovery,
    RoleHypothesis,
)
from can_discovery.relationships.physical.signal_graph import (
    _edge,
    _field_layouts,
    build_signal_graph,
    conditional_field_candidates,
    conditional_layouts,
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
    group = next(item for item in graph['role_groups'] if item['role'] == 'speed')
    assert group['members'] == [0, 1]
    assert group['observed_roots'] == [0]
    assert group['origin'] == 'unknown'
    assert 'rpm' in graph['null_roles']


def test_independent_sensors_remain_distinct_roots():
    captures = _captures(independent=True)
    fields = tuple(_field(arb_id) for arb_id in ('100', '300'))
    discovery = PhysicalDiscovery(tuple(RoleHypothesis(field, 'speed', .9,
        (), ()) for field in fields), (), (), ())
    graph = build_signal_graph(captures, FieldRegistry(fields, fields), discovery)
    groups = [item for item in graph['role_groups'] if item['role'] == 'speed']
    assert len(groups) == 2
    assert {tuple(item['observed_roots']) for item in groups} == {(0,), (1,)}


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


def test_conditional_layouts_preserve_mode_alternatives():
    rng = np.random.default_rng(4)
    captures = []
    samples = []
    for seed in (1, 2):
        frames = []
        payloads = []
        for i in range(240):
            mode = (i // 80) % 2
            value = int(rng.integers(0, 65536))
            payload = bytes([mode]) + value.to_bytes(2, 'little')
            payloads.append(payload)
            frames.append(Frame(i*.02, i, '400', payload))
        captures.append(Capture(str(seed), str(seed), tuple(frames), 0., 4.8,
                                0, 0, {}, {}))
        samples.append(tuple(payloads))
    trace = MessageTrace('400', 3, None, tuple(samples), ('1', '2'))
    registry = FieldRegistry((), (), (tokenize(trace, max_layouts=8),))
    modes = conditional_layouts(registry, captures)
    assert len(modes) == 1
    assert modes[0]['selector_byte'] == 0
    assert {item['value'] for item in modes[0]['modes']} == {0, 1}
    assert all(item['alternatives'] for item in modes[0]['modes'])
    fields = conditional_field_candidates(modes, captures)
    assert fields
    assert {field.mux_value for field in fields} == {0, 1}
    assert all(field.mux_start == 0 for field in fields)
    assert all(field.start >= 8 for field in fields)
    assert all(_field_layouts(list(fields), registry, modes))


def test_signal_manifest_rejects_target_source_capture_overlap(tmp_path):
    from scripts.benchmark_can_signal_graph import _signal_entries
    entries = []
    for vehicle in range(4):
        paths = []
        for route in range(3):
            path = tmp_path / f'{vehicle}-{route}.csv'
            path.write_text('')
            paths.append(str(path))
        dbc = tmp_path / f'{vehicle}.dbc'
        dbc.write_text('')
        entries.append({'vehicle': str(vehicle), 'references': paths[:2],
                        'held_out': paths[2], 'dbc': str(dbc)})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(entries))
    assert len(_signal_entries(manifest, None)) == 4
    entries[1]['references'][0] = entries[0]['references'][0]
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match='overlap'):
        _signal_entries(manifest, None)
    entries[1]['references'][0] = str(tmp_path / '1-0.csv')
    digest = hashlib.sha256(b'').hexdigest()
    for entry in entries:
        entry['routes'] = [{'capture': entry['references'][0],
                            'converted_sha256': digest,
                            'device': entry['vehicle'],
                            'session': entry['vehicle']}]
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match='hashes overlap'):
        _signal_entries(manifest, None)
