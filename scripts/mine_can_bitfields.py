"""Fit bit-field candidates on frozen references, select on calibration, score test."""
from __future__ import annotations

import json
from pathlib import Path

from graphids.core.data.discovery.behavior.bitfields import (
    BitFieldConfig,
    calibrate,
    fit_references,
    score,
)
from graphids.core.data.discovery.behavior.fields import file_hash

SETS = ('set_01', 'set_02', 'set_03', 'set_04')


def run_set(raw: Path, frozen: Path, out: Path, cfg: BitFieldConfig):
    assignment = json.loads((frozen / 'assignment.json').read_text())
    if [len(assignment[role]) for role in ('reference', 'calibration', 'normal_test')] != [2, 1, 1]:
        raise ValueError('expected frozen 2/1/1 assignment')
    for role in ('reference', 'calibration', 'normal_test'):
        for item in assignment[role]:
            if file_hash(raw / item['path']) != item['sha256']:
                raise ValueError(f"frozen assignment hash changed: {item['path']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / 'assignment.json').write_text(json.dumps(assignment, indent=2) + '\n')
    reference = fit_references([raw / item['path'] for item in assignment['reference']], cfg)
    (out / 'reference_candidates.json').write_text(json.dumps(reference, indent=2) + '\n')
    calibrated = calibrate(reference, raw / assignment['calibration'][0]['path'])
    (out / 'profile.json').write_text(json.dumps(calibrated, indent=2) + '\n')
    result = score(raw / assignment['normal_test'][0]['path'], calibrated)
    table = result.pop('table')
    with (out / 'normal_test.jsonl').open('w') as handle:
        for row in table:
            handle.write(json.dumps(row, separators=(',', ':')) + '\n')
    result['table'] = str(out / 'normal_test.jsonl')
    report = {'version': 2, 'set': raw.name, 'assignment': assignment,
              'reference_candidates': len(reference['candidates']),
              'calibration_eligible': calibrated['calibration_eligible_count'],
              'selected_rules': len(calibrated['rules']), 'normal_test': result,
              'notes': ['Reference data sets numeric limits; calibration selects rules.',
                        'Normal test is scored after the rule set is frozen.',
                        'Fields are anonymous statistical candidates, not decoded signals.',
                        'Capture independence remains unverified.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main() -> None:
    """Run the unified version-2 behavior discovery pipeline."""
    from scripts.mine_can_behavior import main as behavior_main

    behavior_main()

if __name__ == '__main__':
    main()
