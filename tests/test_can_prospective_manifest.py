import hashlib
import json
from types import SimpleNamespace

import pytest

from can_discovery.manifests.graph import load_graph_cohort
from can_discovery.relationships.physical.source_calibration import (
    fit_source_factor_validation,
)
from scripts.benchmark_can_candidate_graph import promotion_gate
from scripts.prepare_can_prospective_cohort import distinct_routes


def test_manifest_hash_and_session_separation(tmp_path):
    entries = []
    for vehicle in ('a', 'b', 'c', 'd'):
        dbc = tmp_path / f'{vehicle}.dbc'
        dbc.write_text(vehicle)
        dbc_hash = hashlib.sha256(dbc.read_bytes()).hexdigest()
        routes = []
        for index, role in enumerate(('fit_1', 'fit_2', 'dev_1', 'dev_2', 'final')):
            path = tmp_path / f'{vehicle}_{role}.csv'
            path.write_text(f'{vehicle}:{role}')
            routes.append({'capture': str(path), 'vehicle': vehicle, 'bus': 0,
                           'dataset_revision': 'dataset-pin', 'opendbc_revision': 'dbc-pin',
                           'device': f'{vehicle}{index}',
                           'session': f'{vehicle}s{index}',
                           'compressed_sha256': 'a' * 64,
                           'converted_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'dbc_sha256': dbc_hash})
        entries.append({'vehicle': vehicle, 'fitting': [r['capture'] for r in routes[:2]],
                        'validation': [r['capture'] for r in routes[2:4]],
                        'final': routes[4]['capture'], 'dbc': str(dbc), 'routes': routes})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(entries))
    assert len(load_graph_cohort(path, policy='candidate_graph')) == 4
    entries[0]['routes'][4]['session'] = entries[0]['routes'][0]['session']
    path.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match='distinct sessions'):
        load_graph_cohort(path, policy='candidate_graph')
    entries[0]['routes'][4]['session'] = 'new'
    entries[0]['routes'][4]['converted_sha256'] = 'wrong'
    path.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match='hash changed'):
        load_graph_cohort(path, policy='candidate_graph')


def test_explicit_source_validation_rejects_overlap_and_target():
    capture = SimpleNamespace(sha256='same', frames=())
    with pytest.raises(ValueError, match='independent validation'):
        fit_source_factor_validation([('source', [capture, capture], [capture], None)],
                                     target_vehicle='target')
    with pytest.raises(ValueError, match='target-vehicle DBC'):
        fit_source_factor_validation([('target', [capture, capture], [capture], None)],
                                     target_vehicle='target')
