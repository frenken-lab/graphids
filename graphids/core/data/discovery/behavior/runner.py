"""Frozen-assignment orchestration for standalone discovery pilots."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from graphids.core.data.datasets.can_bus import declared_vehicle_id

from .capture import file_hash, load_capture
from .profile import BehaviorConfig, fit_profile
from .tracker import score_capture


def assign(raw: Path) -> dict:
    normal = sorted((raw / 'train_01_attack_free').glob('*.csv'))
    if len(normal) != 4:
        candidates = sorted(path for folder in raw.glob('train_*')
                            for path in folder.glob('*attack-free*.csv'))
        by_hash: dict[str, list[str]] = {}
        for path in candidates:
            by_hash.setdefault(file_hash(path), []).append(str(path.relative_to(raw)))
        duplicates = [paths for paths in by_hash.values() if len(paths) > 1]
        raise ValueError(
            f'2/1/1 normal-role assignment requires four files under {raw}; '
            f'found {len(normal)} in train_01_attack_free and {len(by_hash)} unique '
            f'normal-file hashes across train directories; duplicate groups: {duplicates}. '
            'Do not reuse duplicate or different-vehicle files as independent references.'
        )
    accessory = [path for path in normal if 'accessory' in path.stem]
    other = [path for path in normal if path not in accessory]
    if len(accessory) == 2 and len(other) == 2:
        roles = {'reference': [accessory[0], other[0]], 'calibration': [accessory[1]],
                 'normal_test': [other[1]]}
    else:
        roles = {'reference': normal[:2], 'calibration': normal[2:3], 'normal_test': normal[3:]}
    assignment = {role: [{'path': str(path.relative_to(raw)), 'sha256': file_hash(path)}
                         for path in paths] for role, paths in roles.items()}
    hashes = [item['sha256'] for entries in assignment.values() for item in entries]
    if len(set(hashes)) != len(hashes):
        raise ValueError('normal roles contain duplicate files')
    return assignment


def _verify(raw: Path, assignment: dict) -> None:
    if [len(assignment[role]) for role in ('reference', 'calibration', 'normal_test')] != [2, 1, 1]:
        raise ValueError('expected frozen 2/1/1 assignment')
    hashes = []
    for role in ('reference', 'calibration', 'normal_test'):
        for item in assignment[role]:
            path = raw / item['path']
            actual = file_hash(path)
            if actual != item['sha256']:
                raise ValueError(f'frozen assignment hash changed: {path}')
            hashes.append(actual)
    if len(set(hashes)) != len(hashes):
        raise ValueError('normal roles contain duplicate files')


def _score(path: Path, profile, destination: Path) -> dict:
    capture = load_capture(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    evidence: Counter[str] = Counter()
    bins = 0
    with destination.open('w') as handle:
        for features, scores in score_capture(capture, profile):
            bins += 1
            values = scores.to_dict()
            evidence.update(key for key, value in values.items() if value > 0)
            handle.write(json.dumps({'features': features.to_dict(), 'scores': values},
                                    separators=(',', ':')) + '\n')
    hours = (capture.end - capture.start) / 3600
    return {'source': capture.source, 'sha256': capture.sha256, 'rows': len(capture.frames),
            'duration_seconds': capture.end - capture.start, 'bins': bins,
            'normal_exposure_hours': hours,
            'evidence_bins': dict(sorted(evidence.items())),
            'evidence_bins_per_hour': {key: count / hours if hours else None
                                       for key, count in sorted(evidence.items())},
            'table': str(destination)}


def run_set(raw: Path, out: Path, *, frozen: Path | None = None,
            config: BehaviorConfig | None = None, include_attacks: bool = True) -> dict:
    """Fit on references, score disjoint roles, and write versioned artifacts."""
    config = config or BehaviorConfig()
    assignment = json.loads((frozen / 'assignment.json').read_text()) if frozen else assign(raw)
    _verify(raw, assignment)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'assignment.json').write_text(json.dumps(assignment, indent=2) + '\n')
    references = [load_capture(raw / item['path']) for item in assignment['reference']]
    profile = fit_profile(references, config)
    (out / 'profile.json').write_text(json.dumps(profile.to_dict(), indent=2, allow_nan=False) + '\n')
    results = {role: [_score(raw / item['path'], profile,
                      out / 'tables' / f'{role}_{Path(item["path"]).stem}.jsonl')
                      for item in assignment[role]] for role in ('calibration', 'normal_test')}
    results['attacks'] = []
    if include_attacks:
        for folder in sorted(raw.glob('test_*')):
            if not folder.is_dir():
                continue
            if folder.name.startswith(('test_01_', 'test_02_', 'test_03_', 'test_04_')):
                if declared_vehicle_id(raw.name, folder.name) != declared_vehicle_id(raw.name, 'train_01_attack_free'):
                    continue
            for path in sorted(folder.rglob('*.csv')):
                label = str(path.relative_to(raw)).replace('/', '__')
                result = _score(path, profile, out / 'tables' / f'{label}.jsonl')
                result['interpretation'] = 'diagnostic_unlabeled' if 'suppress' in folder.name else 'diagnostic'
                results['attacks'].append(result)
    report = {'version': 2, 'set': raw.name, 'assignment': assignment,
              'rule_counts': {'timing': len(profile.timing), 'payload': len(profile.payload),
                              'fields': len(profile.fields), 'relations': len(profile.relations),
                              'motifs': len(profile.motifs)},
              'results': results,
              'notes': ['Reference definitions use only frozen normal references.',
                        'Physical field names are unverified behavioral hypotheses.',
                        'Attack files are scored only after the profile is frozen.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
