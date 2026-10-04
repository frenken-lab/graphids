from __future__ import annotations

import csv
import hashlib
import json

from graphids.core.data.discovery.behavior.fields import FieldConfig, fit, score
from scripts.mine_can_fields import run_set


def write(path, *, jump=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp', 'arbitration_id', 'data_field', 'attack'])
        for i in range(160):
            count = i % 16
            value = i // 2
            if jump and i == 100:
                count = (count + 5) % 16
                value = 220
            writer.writerow([i * .1, 'A', f'{count:01x}0{value:02x}', int(jump and i == 100)])


def test_reference_only_roles_and_scoring(tmp_path):
    paths = [tmp_path / f'ref{i}.csv' for i in range(2)]
    for path in paths:
        write(path)
    cfg = FieldConfig()
    profile = fit(paths, cfg)
    assert any(r['role'] == 'counter_candidate' and r['byte_index'] == 0 for r in profile['rules'])
    assert any(r['role'] == 'smooth_byte' and r['byte_index'] == 1 for r in profile['rules'])
    normal = score(paths[0], profile)
    assert normal['alert_bins'] == {'counter': 0, 'numeric_jump': 0}
    attack = tmp_path / 'attack.csv'
    write(attack, jump=True)
    result = score(attack, profile)
    assert result['alert_bins']['counter'] > 0
    assert result['alert_bins']['numeric_jump'] > 0
    assert result['detected_attack_bins']['counter'] == 1
    assert result['detected_attack_bins']['numeric_jump'] == 1
    assert fit(paths, cfg) == profile


def test_frozen_assignment_and_no_attack_tuning(tmp_path):
    raw = tmp_path / 'set_01'
    names = ['accessory-1', 'attack-free-1', 'accessory-2', 'attack-free-2']
    for name in names:
        write(raw / 'train_01_attack_free' / f'{name}.csv')
    roles = {'reference': names[:2], 'calibration': names[2:3], 'normal_test': names[3:]}
    assignment = {role: [{'path': f'train_01_attack_free/{name}.csv',
                          'sha256': hashlib.sha256((raw / 'train_01_attack_free' / f'{name}.csv').read_bytes()).hexdigest()}
                         for name in values] for role, values in roles.items()}
    frozen = tmp_path / 'pilot'
    frozen.mkdir()
    (frozen / 'assignment.json').write_text(json.dumps(assignment))
    report = run_set(raw, frozen, tmp_path / 'out', FieldConfig(), attacks=False)
    assert report['rule_count'] > 0
    assert report['results']['normal_test']['alert_bins']['counter'] == 0
    (raw / 'train_01_attack_free' / 'accessory-2.csv').write_text('changed')
    try:
        run_set(raw, frozen, tmp_path / 'out2', FieldConfig(), attacks=False)
    except ValueError as exc:
        assert 'hash changed' in str(exc)
    else:
        raise AssertionError('assignment mutation was accepted')
