from pathlib import Path

import pytest

from can_discovery.iterative.rounds import (
    artifact_hash,
    freeze_manifest,
    promote_round,
    verify_manifest,
    write_artifact,
)


def test_manifest_hashes_and_immutable_artifact(tmp_path):
    paths = [tmp_path / str(i) for i in range(4)]
    for i, path in enumerate(paths):
        path.write_text(str(i))
    entry = {'vehicle': 'V', 'references': [str(paths[0]), str(paths[1])],
             'held_out': str(paths[2]), 'dbc': str(paths[3])}
    manifest = freeze_manifest([entry])
    assert manifest['vehicles'][0]['captures']['final_evaluation'] == []
    verify_manifest(manifest)
    output = tmp_path / 'artifact.json'
    assert write_artifact(output, manifest) == artifact_hash(manifest)
    assert write_artifact(output, manifest) == artifact_hash(manifest)
    with pytest.raises(ValueError, match='immutable'):
        write_artifact(output, {'changed': True})
    paths[0].write_text('changed')
    with pytest.raises(ValueError, match='hash changed'):
        verify_manifest(manifest)


def test_promotion_gates_and_cap():
    shared = {'round': 0, 'factor_weights': {'speed_acceleration': .34},
              'source_factors': [{'name': 'speed_acceleration',
                                  'supporting_vehicles': ['A', 'B']}],
              'source_vehicles': ['A', 'B'],
              'source_hashes': {'A': {'captures': ['a']}, 'B': {'captures': ['b']}},
              'promotion_records': []}
    record = {'factor': 'speed_acceleration', 'vehicle': 'C', 'passed': True,
              'validation_hashes': ['c'], 'ancestry': [], 'proposal_rank': 1}
    validation = {'records': [record]}
    promoted = promote_round(shared, [validation])
    assert promoted['factor_weights']['speed_acceleration'] == .35
    assert shared['factor_weights']['speed_acceleration'] == .34
    record['vehicle'] = 'A'
    record['ancestry'] = [artifact_hash(shared)]
    rejected = promote_round(shared, [validation])
    assert rejected['factor_weights']['speed_acceleration'] == .34
    assert rejected['promotion_records'][0]['reason'] == 'circular_source_ancestry'
