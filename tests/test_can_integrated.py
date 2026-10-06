from __future__ import annotations

from dataclasses import replace

import numpy as np

from can_discovery.decoding.signals import FieldCandidate, FieldRegistry
from can_discovery.engineering import integrated
from can_discovery.preprocessing.capture import Capture, Frame


def field(arb_id: str, *, start: int = 0, label: str = 'SPEED') -> FieldCandidate:
    return FieldCandidate(arb_id, start, 8, 'little', False, 'known', 0., 1., (),
                          ((label, 1.),), (), 2)


def capture(name: str, ids: tuple[str, ...]) -> Capture:
    frames = tuple(Frame(i*.1+j*.001, i*len(ids)+j, arb_id,
                         bytes([i % 256, (i//2) % 256]))
                   for i in range(120) for j, arb_id in enumerate(ids))
    return Capture(name, name, frames, 0., 12., 0, 0, {}, {})


def test_frame_match_needs_other_shared_ids_and_keeps_ambiguous_alternatives(monkeypatch):
    target = [capture('target', ('100', '101', '102'))]
    source = capture('source', ('100', '101', '102'))
    monkeypatch.setattr(integrated, 'load_dbc_fields', lambda _: (field('100'),))
    monkeypatch.setattr(integrated, 'dbc_formats', lambda _: {})
    anchors, decisions = integrated.match_frames(target, [('source', [source], 'unused')])
    assert len(anchors) == 1
    assert decisions[0]['status'] == 'anchor'
    collision = capture('collision', ('100',))
    anchors, decisions = integrated.match_frames(target, [('collision', [collision], 'unused')])
    assert not anchors
    assert decisions[0]['status'] == 'alternative'


def test_frame_match_rejects_conflicting_source_layouts(monkeypatch):
    target = [capture('target', ('100', '101', '102'))]
    sources = [('one', [capture('one', ('100', '101', '102'))], 'one.dbc'),
               ('two', [capture('two', ('100', '101', '102'))], 'two.dbc')]
    monkeypatch.setattr(integrated, 'load_dbc_fields', lambda path:
                        (field('100', start=0 if path == 'one.dbc' else 8),))
    monkeypatch.setattr(integrated, 'dbc_formats', lambda _: {})
    anchors, decisions = integrated.match_frames(target, sources)
    assert anchors == ()
    assert decisions[0]['status'] == 'alternative'


def test_matched_anchor_owns_bits_and_preserves_token_alternative():
    anchor = replace(field('100'), bus=None)
    token = replace(field('100', label='unknown'), role='unknown',
                    label_hypotheses=(('unknown', 1.),))
    registry = integrated.merge_registry(FieldRegistry((token,), (token,)), (anchor,))
    assert registry.selected == (anchor,)
    assert len(registry.candidates) == 1  # exact coordinates deduplicated
    assert registry.conflicts[0]['reason'] == 'token_overlap'


def test_different_mux_modes_can_share_bits():
    a = replace(field('100'), mux_start=0, mux_value=1)
    b = replace(field('100'), mux_start=0, mux_value=2)
    registry = integrated.merge_registry(FieldRegistry((), ()), (a, b))
    assert registry.selected == (a, b)


def test_derivative_relation_requires_independent_replication():
    speed = field('100')
    accel = field('200', label='ACCEL')
    captures = []
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        a = rng.integers(0, 3, 500).astype(int)
        v = np.cumsum(a) + 100
        frames = []
        for i, (value, change) in enumerate(zip(v, a)):
            frames.extend((Frame(i*.1, 2*i, '100', int(value).to_bytes(2, 'little')),
                           Frame(i*.1+.001, 2*i+1, '200', int(change).to_bytes(2, 'little'))))
        captures.append(Capture(str(seed), str(seed), tuple(frames), 0., 50.,
                                0, 0, {}, {}))
    result = integrated.derivative_relation(captures[:2], captures[2:], speed,
                                            accel, operation='derivative')
    assert result['status'] in ('supported', 'unsupported')
    assert len(result['routes']) == 3


def test_relation_groups_keep_related_sensors_distinct():
    fields = tuple(field(arb_id) for arb_id in ('100', '200', '300'))
    edges = [{'source': 0, 'target': 1, 'relation': 'affine_copy',
              'lag_seconds': .02},
             {'source': 0, 'target': 2, 'relation': 'transformed_estimate',
              'lag_seconds': .02}]
    groups = integrated.relation_groups(fields, edges)
    assert [group['members'] for group in groups] == [[0, 1], [2]]
    assert groups[0]['observed_roots'] == [0]


def test_wheel_mean_relation_replication():
    fields = tuple(field(arb_id) for arb_id in ('100', '101', '102', '103', '104'))
    captures = []
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        values = np.cumsum(rng.integers(-2, 3, 500)) + 100
        frames = tuple(Frame(i*.1+j*.001, i*5+j, fields[j].arb_id,
                             int(value).to_bytes(2, 'little'))
                       for i, value in enumerate(values) for j in range(5))
        captures.append(Capture(str(seed), str(seed), frames, 0., 50.,
                                0, 0, {}, {}))
    evidence = integrated.wheel_mean_relation(captures[:2], captures[2:],
                                              fields[0], fields[1:])
    assert evidence['status'] == 'supported'


def test_decode_physical_applies_dbc_scale_and_offset():
    data = capture('one', ('100',))
    times, values = integrated.decode_physical(data, field('100'),
                                                {'scale': .5, 'offset': -2})
    assert len(times) == len(values) == 120
    assert values[4] == 0
