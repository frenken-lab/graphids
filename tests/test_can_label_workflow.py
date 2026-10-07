"""Current label-completion stage contracts and sealed-route protection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from can_discovery.decoding.signals import FieldCandidate
from can_discovery.label_completion import current, full_search
from can_discovery.label_completion.manifest import freeze_manifest
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

    monkeypatch.setattr(full_search, 'load_capture', capture)
    discovery_limits = []
    def discover(*args, **kwargs):
        discovery_limits.append(kwargs['max_ids'])
        return SimpleNamespace(candidates=())
    monkeypatch.setattr(full_search, 'discover_fields', discover)
    monkeypatch.setattr(full_search, 'load_dbc_fields', lambda path: ())
    monkeypatch.setattr(full_search, 'rank_all_families', lambda *args: [])
    monkeypatch.setattr(full_search, '_reference_fields', lambda *args: ())
    monkeypatch.setattr(full_search, 'mine_case', lambda *args: {
        'dbc_annotations': {'reference_layouts': []}})
    monkeypatch.setattr(full_search, 'attach_role_ranks', lambda case, fitting: case)
    prepared = full_search.run(manifest_path, output, 0, manifest_path)
    assert prepared['final_routes_opened'] is False
    assert discovery_limits == [1_000_000, 64]
    assert (output / 'v-accelerator_pedal-baseline.json').exists()
    assert (output / 'v-accelerator_pedal-pool.jsonl.gz').exists()

    monkeypatch.setattr(current, 'load_capture', capture)
    ranked = current.run(manifest_path, manifest_path, output, output, output, 0)
    assert ranked['schema'] == SCHEMA
    assert ranked['final_routes_opened'] is False
    assert ranked['review_ledger']['abstained'] is True
    assert opened == routes[:4] * 2

    routes[0].write_text('changed')
    with pytest.raises(ValueError, match='hash changed'):
        full_search.run(manifest_path, output, 0, manifest_path)


def test_staging_cannot_redirect_sealed_final_route(tmp_path):
    manifest_path, _ = _manifest(tmp_path)
    original = json.loads(manifest_path.read_text())
    staged = json.loads(manifest_path.read_text())
    staged['vehicles'][0]['routes']['final'][0]['path'] = '/tmp/other-final'
    with pytest.raises(ValueError, match='sealed final'):
        full_search.verify_staged_manifest(original, staged)


def test_full_search_retains_all_scored_layouts(monkeypatch):
    field = FieldCandidate('100', 0, 8, 'little', False, 'unknown', 0., .5,
                           (100, 100), (('unknown', 1.),), ('test',), 2)
    registry = SimpleNamespace(candidates=(field,))
    monkeypatch.setattr(full_search, '_series', lambda *args: [])
    monkeypatch.setattr(full_search, '_score', lambda candidate, *_: (float(candidate.start), ()))
    group = {'bus': None, 'arb_id': '100', 'payload_length': 2, 'protocol_bits': []}
    scored = full_search.score_family(group, 'vehicle_speed', [], registry, [], max_width=2)
    assert scored['pool_size'] == len(scored['ordered'])
    assert scored['pool_size'] > 8
    assert scored['ordered'][0]['score'] >= scored['ordered'][-1]['score']


def test_report_validates_case_identity_and_summarizes(tmp_path, monkeypatch):
    from can_discovery.label_completion.baseline import ROLES
    from can_discovery.label_completion.manifest import write_artifact
    from can_discovery.label_completion.report import aggregate, report

    manifest_path, _ = _manifest(tmp_path)
    output = tmp_path / 'out'
    monkeypatch.setattr(full_search, 'load_capture', lambda path:
                        SimpleNamespace(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    monkeypatch.setattr(full_search, 'discover_fields', lambda *args, **kwargs:
                        SimpleNamespace(candidates=()))
    monkeypatch.setattr(full_search, 'load_dbc_fields', lambda path: ())
    monkeypatch.setattr(full_search, 'rank_all_families', lambda *args: [])
    monkeypatch.setattr(full_search, '_reference_fields', lambda *args: ())
    monkeypatch.setattr(full_search, 'mine_case', lambda *args: {
        'dbc_annotations': {'reference_layouts': []}})
    monkeypatch.setattr(full_search, 'attach_role_ranks', lambda case, fitting: case)
    monkeypatch.setattr(current, 'load_capture', lambda path:
                        SimpleNamespace(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    monkeypatch.setattr(current, 'load_dbc_fields', lambda path: ())
    for index, _ in enumerate(ROLES):
        full_search.run(manifest_path, output, index, manifest_path)
    for index, _ in enumerate(ROLES):
        current.run(manifest_path, manifest_path, output, output, output, index)
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
