"""Current label-completion stage contracts and sealed-route protection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from can_discovery.label_completion import rank, case
from can_discovery.decoding.signals import FieldCandidate
from can_discovery.scoring import dbc_assisted, role_guided
from can_discovery.manifests.label_completion import freeze_manifest, verify_staged_manifest
from can_discovery.label_completion.progressive_v6 import SCHEMA


def _manifest(tmp_path: Path) -> tuple[Path, list[Path]]:
    routes = [tmp_path / f'{index}.csv' for index in range(5)]
    for index, route in enumerate(routes):
        route.write_text(str(index))
    dbc = tmp_path / 'vehicle.dbc'
    dbc.write_text('test')
    manifest = freeze_manifest([{
        'vehicle': 'V',
        'fitting': [str(path) for path in routes[:2]],
        'development': [str(path) for path in routes[2:4]],
        'final': {'path': str(routes[4]),
                  'sha256': hashlib.sha256(routes[4].read_bytes()).hexdigest()},
        'dbc': str(dbc),
    }])
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path, routes


def test_prepare_then_rank_keeps_final_route_sealed(tmp_path, monkeypatch):
    manifest_path, routes = _manifest(tmp_path)
    output = tmp_path / 'out'
    opened = []

    def capture(path):
        opened.append(Path(path))
        return SimpleNamespace(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())

    monkeypatch.setattr(case, 'load_capture', capture)
    discovery_limits = []
    def discover(*args, **kwargs):
        discovery_limits.append(kwargs['max_ids'])
        return SimpleNamespace(candidates=())
    monkeypatch.setattr(case, 'discover_fields', discover)
    monkeypatch.setattr(case, 'other_role_anchors', lambda *args: ())
    monkeypatch.setattr(case, 'prepare_families', lambda *args, **kwargs: ())
    monkeypatch.setattr(case, '_reference_fields', lambda *args: ())
    monkeypatch.setattr(case, 'mine_case', lambda *args: {
        'dbc_annotations': {'reference_layouts': []}})
    monkeypatch.setattr(case, 'attach_role_ranks', lambda case, fitting: case)
    prepared = case.run(manifest_path, output, 0, manifest_path)
    assert prepared['final_routes_opened'] is False
    assert discovery_limits == [64]
    assert (output / 'v-accelerator_pedal-baseline.json').exists()
    assert (output / 'v-accelerator_pedal-pool.jsonl.gz').exists()

    monkeypatch.setattr(rank, 'load_capture', capture)
    ranked = rank.run(manifest_path, manifest_path, output, output, output, 0)
    assert ranked['schema'] == SCHEMA
    assert ranked['final_routes_opened'] is False
    assert ranked['review_ledger']['abstained'] is True
    assert opened == routes[:4] * 2

    routes[0].write_text('changed')
    with pytest.raises(ValueError, match='hash changed'):
        case.run(manifest_path, output, 0, manifest_path)


def test_staging_cannot_redirect_sealed_final_route(tmp_path):
    manifest_path, _ = _manifest(tmp_path)
    original = json.loads(manifest_path.read_text())
    staged = json.loads(manifest_path.read_text())
    staged['vehicles'][0]['routes']['final'][0]['path'] = '/tmp/other-final'
    with pytest.raises(ValueError, match='sealed final'):
        verify_staged_manifest(original, staged)


def test_unsupervised_track_skips_dbc_fit_and_source_gates(tmp_path, monkeypatch):
    manifest_path, _ = _manifest(tmp_path)
    output = tmp_path / 'unsupervised'
    def capture(path):
        return SimpleNamespace(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(case, 'load_capture', capture)
    monkeypatch.setattr(case, 'prepare_families', lambda *args, **kwargs: ())
    monkeypatch.setattr(case, 'other_role_anchors', lambda *args:
                        pytest.fail('blind fitting opened the DBC'))
    monkeypatch.setattr(case, '_reference_fields', lambda *args: ())
    monkeypatch.setattr(case, 'discover_fields', lambda *args, **kwargs:
                        SimpleNamespace(candidates=()))
    monkeypatch.setattr(case, 'mine_case', lambda *args: {
        'dbc_annotations': {'reference_layouts': []}})
    monkeypatch.setattr(case, 'attach_role_ranks', lambda case, fitting: case)
    prepared = case.run(manifest_path, output, 0, manifest_path,
                               track='unsupervised')
    assert prepared['track'] == 'unsupervised'

    monkeypatch.setattr(rank, 'load_capture', capture)
    monkeypatch.setattr(rank, 'contrast_gate', lambda *args:
                        pytest.fail('blind ranking used source DBC labels'))
    monkeypatch.setattr(rank, 'relation_gate', lambda *args:
                        pytest.fail('blind ranking used a target DBC anchor'))
    ranked = rank.run(manifest_path, manifest_path, output, output, output, 0,
                         track='unsupervised')
    assert ranked['track'] == 'unsupervised'
    assert ranked['source_contrast']['status'] == 'not_applicable'
    with pytest.raises(ValueError, match='track mismatch'):
        rank.run(manifest_path, manifest_path, output, output, output, 0,
                    track='dbc_assisted')


def test_dbc_assisted_filter_is_separate_from_role_guided_scoring():
    from can_discovery.preprocessing.traces import MessageTrace
    from can_discovery.decoding.tokenization import tokenize
    trace = MessageTrace('100', 1, None, ((), ()), ('a', 'b'))
    family = {'bus': None, 'arb_id': '100', 'payload_length': 1}
    anchor = FieldCandidate('100', 0, 1, 'little', False, 'known', 0., 1., (),
                            (('RPM', 1.),), ('dbc',), 1)
    layouts = lambda: tokenize(trace, method='exhaustive', max_width=1).candidates()
    blind = list(layouts())
    assisted = list(dbc_assisted.filter_known_role_anchors(layouts(), family, (anchor,)))
    assert len(blind) == 8
    assert len(assisted) == 7


def test_report_validates_case_identity_and_summarizes(tmp_path, monkeypatch):
    from can_discovery.label_completion.baseline import ROLES
    from can_discovery.manifests.label_completion import write_artifact
    from can_discovery.label_completion.report import aggregate, report

    manifest_path, _ = _manifest(tmp_path)
    output = tmp_path / 'out'
    monkeypatch.setattr(case, 'load_capture', lambda path:
                        SimpleNamespace(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    monkeypatch.setattr(case, 'discover_fields', lambda *args, **kwargs:
                        SimpleNamespace(candidates=()))
    monkeypatch.setattr(case, 'other_role_anchors', lambda *args: ())
    monkeypatch.setattr(case, 'prepare_families', lambda *args, **kwargs: ())
    monkeypatch.setattr(case, '_reference_fields', lambda *args: ())
    monkeypatch.setattr(case, 'mine_case', lambda *args: {
        'dbc_annotations': {'reference_layouts': []}})
    monkeypatch.setattr(case, 'attach_role_ranks', lambda case, fitting: case)
    monkeypatch.setattr(rank, 'load_capture', lambda path:
                        SimpleNamespace(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    monkeypatch.setattr(dbc_assisted, 'load_dbc_fields', lambda path: ())
    for index, _ in enumerate(ROLES):
        case.run(manifest_path, output, index, manifest_path)
    for index, _ in enumerate(ROLES):
        rank.run(manifest_path, manifest_path, output, output, output, index)
    summary = aggregate(manifest_path, manifest_path, output)
    assert len(summary['cases']) == len(ROLES)
    assert summary['final_routes_opened'] is False
    assert summary['metrics']['layouts_scored'] == 0
    assert f'0 / {len(ROLES)}' in report(summary)
    write_artifact(output / 'summary.json', summary)

    case_path = output / f'v-{ROLES[0]}-case.json'
    tampered = json.loads(case_path.read_text())
    tampered['role'] = ROLES[1]
    case_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match='provenance mismatch'):
        aggregate(manifest_path, manifest_path, output)
    tampered['role'] = ROLES[0]
    tampered['track'] = 'unsupervised'
    case_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match='provenance mismatch'):
        aggregate(manifest_path, manifest_path, output)
