from __future__ import annotations

import csv
import hashlib
import json

from graphids.core.data.discovery.behavior.bitfields import (
    BitFieldConfig,
    calibrate,
    extract,
    fit_references,
    score,
)
from scripts.mine_can_bitfields import run_set


def write(path, *, attacked=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp', 'arbitration_id', 'data_field', 'attack'])
        for i in range(160):
            value = 1000 + i // 2
            if attacked and i == 100:
                value = 3000
            # A 16-bit little-endian value is preceded by a constant byte.
            payload = bytes([0xAA]) + value.to_bytes(2, 'little') + bytes([0x55])
            writer.writerow([i * .1, 'A', payload.hex(), int(attacked and i == 100)])


def test_extract_endian_signed():
    payload = bytes.fromhex('ff7f')
    assert extract(payload, 0, 16, 'little', False) == 32767
    assert extract(bytes.fromhex('ff80'), 0, 16, 'big', True) == -128
    assert extract(payload, 8, 12, 'big', False) is None


def test_frozen_calibration_and_scoring(tmp_path):
    paths = [tmp_path / f'ref{i}.csv' for i in range(2)]
    for path in paths:
        write(path)
    calibration = tmp_path / 'calibration.csv'
    normal = tmp_path / 'normal.csv'
    attack = tmp_path / 'attack.csv'
    for path in (calibration, normal):
        write(path)
    write(attack, attacked=True)
    reference = fit_references(paths, BitFieldConfig())
    profile = calibrate(reference, calibration)
    assert profile['candidate_count'] > 0
    assert profile['rules']
    assert score(normal, profile)['alert_bins'] == 0
    result = score(attack, profile)
    assert result['alert_bins'] > 0
    assert result['detected_attack_bins'] > 0
    assert profile['calibration_hash'] == hashlib.sha256(calibration.read_bytes()).hexdigest()


def test_late_reference_jump_rejects_early_candidate(tmp_path):
    clean = tmp_path / 'clean.csv'
    unstable = tmp_path / 'unstable.csv'
    write(clean)
    write(unstable, attacked=True)
    profile = fit_references([clean, unstable], BitFieldConfig(samples_per_id=80))
    assert profile['initial_candidate_count'] > len(profile['candidates'])


def test_runner_rejects_changed_normal_file(tmp_path):
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
    report = run_set(raw, frozen, tmp_path / 'out', BitFieldConfig())
    assert report['selected_rules'] > 0
    assert report['normal_test']['alert_bins'] == 0
    assert report['normal_test']['covered_fraction'] == 1
    (raw / 'train_01_attack_free' / 'accessory-2.csv').write_text('changed')
    try:
        run_set(raw, frozen, tmp_path / 'out2', BitFieldConfig())
    except ValueError as exc:
        assert 'hash changed' in str(exc)
    else:
        raise AssertionError('assignment mutation was accepted')
