"""Controlled retrieval and source-gate checks."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from can_discovery.decoding.signals import FieldCandidate
from can_discovery.label_completion import controlled_v3 as v3
from can_discovery.preprocessing.capture import Capture, Frame
from can_discovery.label_completion.manifest import artifact_hash, freeze_manifest


def _field(arb_id: str, start: int = 0):
    return FieldCandidate(arb_id, start, 16, 'little', False, 'unknown',
                          0., .5, (200, 200), (('unknown', 1.),),
                          ('synthetic',), 4)


def test_adaptive_budget_uses_only_rank_margin():
    families = [{'score': 1 - i*.01} for i in range(20)]
    assert v3.adaptive_limit(families) == 16
    families = [{'score': 1 - i*.04} for i in range(20)]
    assert v3.adaptive_limit(families) == 8


def test_relation_route_has_shifted_control_and_requires_change():
    rng = np.random.default_rng(4)
    steps = rng.normal(size=300)
    speed = np.cumsum(steps) + 100
    rpm = 2*speed + rng.normal(scale=.1, size=300)
    frames = []
    for i in range(300):
        frames.append(Frame(i*.1, 2*i, '100',
                            int(speed[i]*10).to_bytes(4, 'little', signed=False)))
        frames.append(Frame(i*.1, 2*i+1, '200',
                            int(rpm[i]*10).to_bytes(4, 'little', signed=False)))
    capture = Capture('route', 'hash', tuple(frames), 0., 29.9, 0, 0, {}, {})
    row = v3.relation_route(capture, _field('100'), _field('200'))
    assert row['status'] == 'measured'
    assert row['actual'] > row['shifted_control']
    assert row['net'] > .1


def test_source_gate_excludes_target_and_needs_two_sources(monkeypatch):
    field = _field('100')
    monkeypatch.setattr(v3, 'load_dbc_fields', lambda path: (field,))
    monkeypatch.setattr(v3, 'source_pair', lambda fields, fitting: (field, field))
    monkeypatch.setattr(v3, 'relation_route', lambda capture, speed, rpm:
                        {'status': 'measured', 'net': .2})
    sources = {name: ([object(), object()], [object(), object()], Path(name))
               for name in ('target', 'source_1', 'source_2')}
    gate = v3.source_gate('target', sources)
    assert gate['status'] == 'eligible'
    assert gate['supporting_vehicles'] == ['source_1', 'source_2']
    assert all(row['vehicle'] != 'target' for row in gate['vehicles'])
    assert v3.source_gate('source_1', sources)['status'] == 'eligible'
    del sources['source_2']
    assert v3.source_gate('target', sources)['status'] == 'rejected'


def test_rejected_relation_keeps_rank_and_pool():
    field = _field('100')
    group = {'bus': None, 'arb_id': '100', 'payload_length': 4,
             'score': .5, 'candidate_layouts': [
                 {'rank': 1, 'score': .5, 'layout': asdict(field)}]}
    groups, rows = v3.rerank([group], [group], [], field, 'engine_rpm', False)
    assert groups == [group]
    assert rows == [group]


def test_equal_budget_variants_do_not_use_hidden_reference_for_allocation(monkeypatch):
    from types import SimpleNamespace

    families = [{'bus': None, 'arb_id': f'{i:X}', 'payload_length': 4,
                 'score': 1-i*.01, 'rank': i+1} for i in range(20)]
    groups = [{**row, 'candidate_layouts': []} for row in families[:8]]
    base = {'vehicle': 'V', 'hidden_role': 'engine_rpm',
            'message_family_hypotheses': families,
            'detailed_families': groups,
            'dbc_annotations': {'reference_layouts': []},
            'metrics': {'exact_layout_pool_coverage': False}}
    monkeypatch.setattr(v3, 'load_dbc_fields', lambda path: ())
    monkeypatch.setattr(v3, 'detail_family',
                        lambda row, *args: {**row, 'candidate_layouts': [],
                                            'protocol_bits': []})
    result = v3.run_case(base, [], [], Path('unused.dbc'),
                         SimpleNamespace(candidates=()),
                         {'status': 'eligible', 'supporting_vehicles': ['A', 'B']})
    ids = result['detailed_family_ids']
    assert result['adaptive_budget'] == 16
    assert all(len(ids[name]) == 16 for name in
               ('expanded_search', 'relation_fixed_pool', 'relation_with_reopening'))
    assert ids['expanded_search'] == ids['relation_fixed_pool'] == ids['relation_with_reopening']
    assert result['relation_gate']['status'] == 'unavailable'


def test_v3_runner_keeps_final_route_sealed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import run_can_label_completion_v3 as runner
    from can_discovery.label_completion.pipeline import ROLES

    routes = [tmp_path / f'{i}.csv' for i in range(5)]
    for i, path in enumerate(routes):
        path.write_text(str(i))
    dbc = tmp_path / 'vehicle.dbc'
    dbc.write_text('dbc')
    manifest = freeze_manifest([{'vehicle': 'V',
        'fitting': [str(x) for x in routes[:2]],
        'development': [str(x) for x in routes[2:4]],
        'final': {'path': str(routes[4]), 'sha256': hashlib.sha256(
            routes[4].read_bytes()).hexdigest()}, 'dbc': str(dbc)}])
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    baseline_dir = tmp_path / 'baseline'
    baseline_dir.mkdir()
    rows = []
    for role in ROLES:
        path = baseline_dir / f'{role}.json'
        path.write_text(json.dumps({'manifest_hash': artifact_hash(manifest)}))
        rows.append({'vehicle': 'V', 'role': role, 'path': str(path)})
    (baseline_dir / 'summary-v2.1.json').write_text(json.dumps({
        'manifest_hash': artifact_hash(manifest), 'cases': rows}))
    opened = []
    monkeypatch.setattr(runner, 'load_capture', lambda path: (
        opened.append(str(path)) or SimpleNamespace(sha256=str(path))))
    monkeypatch.setattr(runner, 'discover_fields', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, 'source_gate', lambda *args: {
        'status': 'rejected', 'supporting_vehicles': []})
    metric = {'id_top_1': False, 'id_top_5': False, 'id_top_8': False,
              'exact_layout_top_1': False, 'exact_layout_top_8': False,
              'held_out_affine_raw_agreement': False}
    monkeypatch.setattr(runner, 'run_case', lambda base, *args: {
        'assessable': False, 'adaptive_budget': 8,
        'relation_gate': {'status': 'unavailable'},
        'variants': {name: metric for name in (
            'baseline', 'expanded_search', 'relation_fixed_pool',
            'relation_with_reopening')}, 'transition_trace': {}})
    runner.run(manifest_path, baseline_dir, tmp_path / 'output')
    assert opened == [str(path) for path in routes[:4]]
