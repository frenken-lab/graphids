"""Structural and leakage guards for the ID-first ledger."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import pytest

from can_discovery.decoding.signals import FieldCandidate
from can_discovery.label_completion.manifest import freeze_manifest, verify_manifest
from can_discovery.label_completion.pipeline import (compatible_layouts,
    enumerate_layouts, interpretation_key, layout_key, rank_families)
from can_discovery.decoding.field_values import extract
from can_discovery.decoding.signals import _bit_positions


def field(start=0, width=8, endian='little', signed=False):
    return FieldCandidate('100', start, width, endian, signed, 'unknown', 0., .5,
                          (100, 100), (('unknown', 1.),), ('test',), 2)


def test_family_score_uses_best_layout_and_deterministic_ties():
    first = field()
    second = field(8)
    rows = [(first, .5), (second, .5), (first, .7)]
    assert rank_families(rows)[0]['score'] == .7
    assert rank_families(rows) == rank_families(list(reversed(rows)))


def test_enumeration_covers_both_orders_and_signedness_and_excludes_protocol():
    rows = list(enumerate_layouts(field(), set()))
    assert layout_key(field(0, 12, 'little', True)) in {layout_key(f) for f in rows}
    assert layout_key(field(0, 12, 'big', False)) in {layout_key(f) for f in rows}
    assert not any(f.signed for f in rows if f.width == 1)
    guarded = list(enumerate_layouts(field(), set(range(8))))
    assert all(not (set(range(f.start, f.start + f.width)) & set(range(8)))
               for f in guarded if f.endian == 'little')
    assert _bit_positions(0, 8, 'big') == _bit_positions(0, 8, 'little')
    assert extract(b'\xff\x00', 0, 8, 'big', True) == -1
    assert extract(b'\xff\x00', 0, 8, 'little', False) == 255
    assert interpretation_key(field(0, 1, 'little')) == interpretation_key(
        field(7, 1, 'big'))


def test_layout_alternatives_preserve_overlapping_candidates():
    rows = [{'field': field(0), 'score': .9},
            {'field': field(4), 'score': .8},
            {'field': field(8), 'score': .7}]
    assert [r['field'].start for r in compatible_layouts(rows)] == [0, 4, 8]


def test_manifest_has_five_distinct_pinned_routes_and_seals_final(tmp_path, monkeypatch):
    from can_discovery.label_completion import manifest as manifest_module
    paths = [tmp_path / f'{i}.csv' for i in range(5)]
    for i, path in enumerate(paths):
        path.write_text(str(i))
    dbc = tmp_path / 'x.dbc'
    dbc.write_text('dbc')
    entry = {'vehicle': 'V', 'fitting': list(map(str, paths[:2])),
             'development': list(map(str, paths[2:4])),
             'final': {'path': str(paths[4]), 'sha256': hashlib.sha256(paths[4].read_bytes()).hexdigest()},
             'dbc': str(dbc)}
    hashed = []
    original_hash = manifest_module.file_hash
    def recording_hash(path):
        hashed.append(str(path))
        return original_hash(path)
    monkeypatch.setattr(manifest_module, 'file_hash', recording_hash)
    manifest = freeze_manifest([entry])
    assert str(paths[4]) not in hashed
    paths[4].write_text('changed')
    verify_manifest(manifest)
    with pytest.raises(ValueError, match='hash changed'):
        verify_manifest(manifest, include_final=True)
    entry['final'] = {'path': str(paths[0]), 'sha256': hashlib.sha256(paths[0].read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match='reused'):
        freeze_manifest([entry])


def test_hidden_dbc_annotation_is_not_an_anchor(monkeypatch):
    from can_discovery.label_completion import pipeline
    from can_discovery.preprocessing.capture import Capture, Frame
    from can_discovery.decoding.signals import FieldRegistry

    hidden = FieldCandidate('100', 0, 8, 'little', False, 'known', 0., 1.,
                            (), (('GAS_PEDAL', 1.),), ('dbc',), 2)
    candidate = field()
    frames = tuple(Frame(float(i), i, '100', bytes((i % 100, 0))) for i in range(40))
    fitting = [Capture(str(i), str(i), frames, 0., 39., 0, 0, {}, {}) for i in range(2)]
    development = [Capture(str(i), str(i), frames, 0., 39., 0, 0, {}, {}) for i in range(2, 4)]
    monkeypatch.setattr(pipeline, 'load_dbc_fields', lambda path: (hidden,))
    monkeypatch.setattr(pipeline, '_reference_fields', lambda *args: ())
    monkeypatch.setattr(pipeline, '_score', lambda *args: (.5, ()))
    monkeypatch.setattr(pipeline, 'enumerate_layouts', lambda *args: iter(()))
    result = pipeline.mine_case('V', 'accelerator_pedal', fitting, development,
                                Path('unused.dbc'), FieldRegistry((candidate,), ()))
    assert result['message_family_hypotheses'][0]['arb_id'] == '100'
    assert result['detailed_families'][0]['dbc_anchor_conflicts'] == []
    assert result['detailed_families'][0]['null_layout']['rank'] is None
    assert result['metrics']['abstained']


def test_other_role_anchor_blocks_overlapping_layout(monkeypatch):
    from can_discovery.label_completion import pipeline
    from can_discovery.preprocessing.capture import Capture, Frame
    from can_discovery.decoding.signals import FieldRegistry

    brake = FieldCandidate('100', 0, 8, 'little', False, 'known', 0., 1.,
                           (), (('BRAKE_PEDAL', 1.),), ('dbc',), 2)
    frames = tuple(Frame(float(i), i, '100', bytes((i % 100, 0))) for i in range(40))
    fitting = [Capture(str(i), str(i), frames, 0., 39., 0, 0, {}, {}) for i in range(2)]
    monkeypatch.setattr(pipeline, 'load_dbc_fields', lambda path: (brake,))
    monkeypatch.setattr(pipeline, '_reference_fields', lambda *args: ())
    result = pipeline.mine_case('V', 'accelerator_pedal', fitting, fitting,
                                Path('unused.dbc'), FieldRegistry((field(),), ()))
    assert result['message_family_hypotheses'] == []


def test_development_runner_does_not_open_final_route(tmp_path, monkeypatch):
    from scripts import run_can_label_completion as runner
    paths = [tmp_path / f'{i}.csv' for i in range(5)]
    for i, path in enumerate(paths):
        path.write_text(str(i))
    dbc = tmp_path / 'x.dbc'
    dbc.write_text('dbc')
    manifest = freeze_manifest([{'vehicle': 'V',
        'fitting': [str(p) for p in paths[:2]],
        'development': [str(p) for p in paths[2:4]],
        'final': {'path': str(paths[4]), 'sha256': hashlib.sha256(paths[4].read_bytes()).hexdigest()},
        'dbc': str(dbc)}])
    manifest_path = tmp_path / 'manifest.json'
    import json
    manifest_path.write_text(json.dumps(manifest))
    opened = []

    def load(path):
        opened.append(str(path))
        return type('CaptureStub', (), {'sha256': str(path)})()

    def mine(vehicle, role, fitting, development, dbc_path, registry):
        return {'hidden_role': role, 'detailed_families': [],
                'metrics': {'id_top_1': False, 'id_top_5': False,
                'id_top_8': False, 'exact_layout_top_1': False,
                'exact_layout_top_8': False, 'exact_layout_pool_coverage': False,
                'held_out_raw_value_agreement': False},
                'dbc_annotations': {'assessable': False, 'reference_layouts': []}}

    monkeypatch.setattr(runner, 'load_capture', load)
    monkeypatch.setattr(runner, 'discover_fields', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, 'mine_case', mine)
    runner.run(manifest_path, tmp_path / 'output')
    assert opened == [str(p) for p in paths[:4]]


def test_final_freeze_refuses_failed_development_gate(tmp_path):
    from scripts.assess_can_label_completion_final import freeze
    import json

    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema': 'dbc-label-completion-manifest-v2',
                                    'vehicles': []}))
    development = tmp_path / 'development'
    development.mkdir()
    (development / 'summary-v2.1.json').write_text(json.dumps({
        'final_assessment_allowed': False}))
    with pytest.raises(ValueError, match='development gate failed'):
        freeze(manifest, development, tmp_path / 'freeze.json')
    assert not (tmp_path / 'freeze.json').exists()


def test_role_rank_is_separate_from_layout_rank(monkeypatch):
    from can_discovery.label_completion import pipeline
    from can_discovery.preprocessing.capture import Capture, Frame
    from can_discovery.label_completion.pipeline import ROLES

    one = field()
    case = {'hidden_role': 'accelerator_pedal',
            'detailed_families': [{'bus': None, 'arb_id': '100', 'payload_length': 2,
                'candidate_layouts': [{'layout': asdict(one), 'rank': 1}]}],
            'dbc_annotations': {'reference_layouts': [asdict(one)]},
            'metrics': {}}
    capture = Capture('fit', 'hash',
        tuple(Frame(float(i), i, '100', b'\x01\x02') for i in range(40)),
        0., 39., 0, 0, {}, {})
    monkeypatch.setattr(pipeline, '_score', lambda field, samples, role:
                        (float(ROLES.index(role)), ()))
    result = pipeline.attach_role_ranks(case, [capture, capture])
    assert result['detailed_families'][0]['candidate_layouts'][0]['rank'] == 1
    assert result['metrics']['role_rank'] == 5
