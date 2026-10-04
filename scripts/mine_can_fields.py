"""Fit anonymous CAN payload roles on frozen normal references and score held-out files."""
from __future__ import annotations

import json
from pathlib import Path

from graphids.core.data.datasets.can_bus import declared_vehicle_id
from graphids.core.data.discovery.behavior.fields import (
    FieldConfig,
    file_hash,
    fit,
    score,
)

SETS = ('set_01', 'set_02', 'set_03', 'set_04')


def run_set(raw: Path, frozen: Path, out: Path, cfg: FieldConfig, attacks: bool = True):
    assignment = json.loads((frozen / 'assignment.json').read_text())
    if [len(assignment[role]) for role in ('reference', 'calibration', 'normal_test')] != [2, 1, 1]:
        raise ValueError('expected frozen 2/1/1 assignment')
    for role in ('reference', 'calibration', 'normal_test'):
        for item in assignment[role]:
            if file_hash(raw / item['path']) != item['sha256']:
                raise ValueError(f"frozen assignment hash changed: {item['path']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / 'assignment.json').write_text(json.dumps(assignment, indent=2) + '\n')
    profile = fit([raw / item['path'] for item in assignment['reference']], cfg)
    (out / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    results = {}
    for role in ('calibration', 'normal_test'):
        result = score(raw / assignment[role][0]['path'], profile)
        table = result.pop('table')
        result['table'] = str(out / f'{role}.jsonl')
        with (out / f'{role}.jsonl').open('w') as handle:
            for row in table:
                handle.write(json.dumps(row, separators=(',', ':')) + '\n')
        results[role] = result
    results['attacks'] = []
    if attacks:
        for folder in sorted(raw.glob('test_*')):
            if not folder.is_dir() or folder.name.startswith('test_05_suppress'):
                continue
            if folder.name.startswith(('test_01_', 'test_02_', 'test_03_', 'test_04_')):
                if declared_vehicle_id(raw.name, folder.name) != declared_vehicle_id(raw.name, 'train_01_attack_free'):
                    continue
            for path in sorted(folder.rglob('*.csv')):
                result = score(path, profile)
                result.pop('table')
                result['family'] = path.stem.rsplit('-', 1)[0]
                result['interpretation'] = 'diagnostic' if 'masquerade' in folder.name else 'frame_labels'
                results['attacks'].append(result)
    report = {'version': 1, 'set': raw.name, 'assignment': assignment,
              'inventory': profile['inventory'], 'rule_count': len(profile['rules']),
              'results': results,
              'notes': ['Rules fit on frozen references only; no threshold fitted on held-out files.',
                        'Byte and nibble roles are anonymous, not decoded physical signals.',
                        'Alert counts are bins; attack-bin recall is not episode recall.',
                        'Normal capture independence is unverified.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main() -> None:
    """Run the unified version-2 behavior discovery pipeline."""
    from scripts.mine_can_behavior import main as behavior_main

    behavior_main()

if __name__ == '__main__':
    main()
