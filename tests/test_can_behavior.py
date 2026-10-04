from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from graphids.core.data.discovery.behavior import (
    BehaviorConfig,
    BehaviorProfile,
    BehaviorState,
    BehaviorTracker,
    fit_profile,
    load_capture,
    score_capture,
)
from graphids.core.data.discovery.behavior.dynamics import DynamicsConfig
from graphids.core.data.discovery.behavior.motifs import (
    MotifRule,
    MotifState,
    MotifTracker,
)
from graphids.core.data.discovery.behavior.runner import assign, run_set


def _write(path: Path, shift: int = 0, *, attacked: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp', 'arbitration_id', 'data_field', 'attack'])
        for index in range(40):
            writer.writerow([index * .1, 'A', f'{(index + shift) % 256:02x}', 0])
            writer.writerow([index * .1 + .02, 'B',
                             f'{(index + shift + (100 if attacked and index == 25 else 1)) % 256:02x}',
                             int(attacked and index == 25)])


def _config():
    return BehaviorConfig(max_nodes=4, max_relations=8, max_motifs=8,
                          motif_min_support=3, field_min_changes=3,
                          payload_min_transitions=3,
                          dynamics=DynamicsConfig(min_id_samples=3, min_pair_samples=3,
                                                  max_candidate_ids=2, max_pair_lag=.08,
                                                  sample_per_stream=100, min_control_gain=.1))


def test_profile_is_reference_only_versioned_and_graph_ready(tmp_path):
    paths = [tmp_path / 'ref1.csv', tmp_path / 'ref2.csv']
    _write(paths[0])
    _write(paths[1], 1)
    refs = [load_capture(path) for path in paths]
    profile = fit_profile(refs, _config())
    assert profile.node_keys == ('A', 'B')
    assert profile.payload and profile.fields and profile.relations
    assert BehaviorProfile.from_dict(json.loads(json.dumps(profile.to_dict()))).to_dict() == profile.to_dict()
    features, scores = next(score_capture(refs[0], profile))
    assert len(features.node) == profile.config.max_nodes
    assert len(features.relation) == profile.config.max_relations
    assert len(features.motif) == profile.config.max_motifs
    assert features.node_mask == (True, True, False, False)
    assert scores.to_dict().keys() == {'timing', 'payload', 'field', 'relation', 'motif'}
    attack = tmp_path / 'attack.csv'
    _write(attack, 1, attacked=True)
    assert fit_profile(refs, _config()).to_dict() == profile.to_dict()
    assert list(score_capture(load_capture(attack), profile))[0] == list(score_capture(refs[1], profile))[0]
    with pytest.raises(ValueError, match='attack'):
        fit_profile([refs[0], load_capture(attack)], _config())


def test_empty_bins_deadline_and_reset(tmp_path):
    paths = [tmp_path / 'ref1.csv', tmp_path / 'ref2.csv']
    _write(paths[0])
    _write(paths[1], 1)
    profile = fit_profile([load_capture(path) for path in paths], _config())
    state = BehaviorState()
    first = load_capture(paths[0]).frames[:2]
    _, _, state = BehaviorTracker.step(first, .05, profile, state)
    features, _, state = BehaviorTracker.step([], .5, profile, state)
    assert features.node[0][2] > 0
    with pytest.raises(ValueError, match='reset'):
        BehaviorTracker.step([], .4, profile, state)
    BehaviorTracker.step(first, .05, profile, BehaviorState())


def test_motif_fork_constraints_and_expiry():
    rule = MotifRule('fork', ('A|change', 'B|change', 'C|change'),
                     ((0, 1, 0, .1), (0, 2, 0, .15)), .2, (10, 10), .9, .1)
    state = MotifState()
    completed, overdue, _, state = MotifTracker.step(
        [(0, 0, 'A', 'change'), (.04, 1, 'B', 'change'), (.09, 2, 'C', 'change')],
        .1, (rule,), state)
    assert completed == [1] and overdue == [0]
    _, _, _, state = MotifTracker.step([(.12, 3, 'A', 'change')], .2, (rule,), state)
    completed, overdue, _, state = MotifTracker.step([], .3, (rule,), state)
    assert completed == [0] and overdue == [1]


def test_runner_verifies_frozen_assignment_and_writes_v2(tmp_path):
    raw = tmp_path / 'set_01'
    for shift, name in enumerate(('accessory-1', 'accessory-2', 'attack-free-1', 'attack-free-2')):
        _write(raw / 'train_01_attack_free' / f'{name}.csv', shift)
    frozen = tmp_path / 'frozen'
    frozen.mkdir()
    (frozen / 'assignment.json').write_text(json.dumps(assign(raw)))
    report = run_set(raw, tmp_path / 'out', frozen=frozen, config=_config(), include_attacks=False)
    assert report['version'] == 2
    assert (tmp_path / 'out' / 'profile.json').exists()
    assert report['results']['normal_test'][0]['bins'] > 0
    test_path = raw / report['assignment']['normal_test'][0]['path']
    test_path.write_text('changed')
    with pytest.raises(ValueError, match='hash changed'):
        run_set(raw, tmp_path / 'out2', frozen=frozen, config=_config(), include_attacks=False)


def test_runner_rejects_duplicate_normal_files_across_train_dirs(tmp_path):
    raw = tmp_path / 'hcrl_ch'
    first = raw / 'train_01_attack_free' / 'attack-free.csv'
    duplicate = raw / 'train_02_with_attacks' / 'attack-free.csv'
    _write(first)
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match='1 unique normal-file hashes'):
        assign(raw)


def test_motif_root_sampling_does_not_alias_one_periodic_phase():
    from graphids.core.data.discovery.behavior.motifs import _count

    roots = [float(index) for index in range(100)]
    targets = [float(index) + .01 for index in range(100) if index % 5 == 1]
    result = _count({'A|arrival': roots, 'B|arrival': targets},
                    ('A|arrival', 'B|arrival'), ((0, 1, 0, .02),), .1, 20)
    assert 0 < result[0] < result[1]
    assert result == _count({'A|arrival': roots, 'B|arrival': targets},
                            ('A|arrival', 'B|arrival'), ((0, 1, 0, .02),), .1, 20)
