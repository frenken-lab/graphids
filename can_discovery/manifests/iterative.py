"""Pinned captures and DBCs for iterative CAN rounds."""
from __future__ import annotations

from pathlib import Path
from .common import file_hash

SCHEMA = 'can-iterative-v1'

def _sha(path: Path) -> str:
    return file_hash(path)


def freeze_manifest(entries: list[dict]) -> dict:
    """Pin captures and DBCs; existing third trips are development validation."""
    frozen, used = [], set()
    for entry in entries:
        vehicle = entry['vehicle']
        fitting = entry.get('fitting', entry.get('references', []))
        validation = entry.get('development_validation', entry.get('validation',
                         [entry['held_out']] if 'held_out' in entry else []))
        if len(fitting) < 2 or not validation:
            raise ValueError(f'{vehicle}: two fitting and one validation capture required')
        captures = {}
        for split, paths in [('fitting', fitting), ('development_validation', validation),
                             ('final_evaluation', entry.get('final_evaluation', []))]:
            captures[split] = []
            for name in paths:
                path = Path(name).resolve()
                digest = _sha(path)
                if digest in used:
                    raise ValueError('capture reused across manifest splits')
                used.add(digest)
                captures[split].append({'path': str(path), 'sha256': digest})
        dbc = Path(entry['dbc']).resolve()
        frozen.append({'vehicle': vehicle, 'captures': captures,
                       'dbc': {'path': str(dbc), 'sha256': _sha(dbc)}})
    return {'schema': SCHEMA, 'kind': 'manifest', 'vehicles': frozen,
            'final_cohort_qualified': False}


def verify_manifest(manifest: dict) -> None:
    if manifest['schema'] != SCHEMA:
        raise ValueError('manifest schema mismatch')
    for entry in manifest['vehicles']:
        for item in [entry['dbc'], *(capture for split in entry['captures'].values()
                                      for capture in split)]:
            if _sha(Path(item['path'])) != item['sha256']:
                raise ValueError(f"input hash changed: {item['path']}")
