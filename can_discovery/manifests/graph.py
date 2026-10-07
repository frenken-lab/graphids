"""Shared cohort parsing for CAN graph and integrated benchmarks.

The candidate graph's prospective provenance checks are stricter than the
signal and integrated benchmarks' historical manifests. Keep that distinction
explicit while sharing split, file, and overlap validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

CohortPolicy = Literal['candidate_graph', 'signal_graph']


def split_paths(entry: dict, split: Literal['fitting', 'validation', 'final']) -> list[str]:
    if split == 'fitting':
        return list(entry.get('fitting', entry.get('references', ())))
    if split == 'validation':
        return list(entry.get('validation', [entry['held_out']] if 'held_out' in entry else ()))
    final = entry.get('final', entry.get('held_out'))
    return [final] if final else []


def load_graph_cohort(manifest: Path, forester_dir: Path | None = None, *,
                      policy: CohortPolicy = 'signal_graph') -> list[dict]:
    """Load four-or-more vehicles with explicit split and provenance checks.

    Route hashes are checked when route records are declared. The candidate
    graph additionally requires its pinned prospective metadata and ordering.
    """
    if policy not in ('candidate_graph', 'signal_graph'):
        raise ValueError(f'unknown cohort policy: {policy}')
    entries = json.loads(Path(manifest).read_text())
    if not isinstance(entries, list):
        raise ValueError('benchmark manifest must be a vehicle list')
    if forester_dir is not None:
        entries.append({'vehicle': 'SUBARU_FORESTER',
                        'references': [str(forester_dir / 'reference_1.csv'),
                                       str(forester_dir / 'reference_2.csv')],
                        'held_out': str(forester_dir / 'reference_3.csv'),
                        'dbc': str(forester_dir / 'subaru_global_2017.dbc')})
    if len(entries) < 4 or len({item['vehicle'].casefold() for item in entries}) != len(entries):
        raise ValueError('benchmark requires at least four distinct vehicles')
    all_paths: list[str] = []
    route_hashes: list[str] = []
    for entry in entries:
        fitting = split_paths(entry, 'fitting')
        validation = split_paths(entry, 'validation')
        final = split_paths(entry, 'final')
        if len(fitting) != 2 or not validation or len(final) != 1:
            raise ValueError('each vehicle needs two fitting, validation, and final captures')
        explicit = any(key in entry for key in ('fitting', 'validation', 'final'))
        if explicit and len(set((*fitting, *validation, *final))) != len(fitting) + len(validation) + 1:
            raise ValueError('capture roles must be disjoint')
        paths = [*fitting, *validation, *final]
        all_paths.extend(set(paths))
        for name in (*paths, entry['dbc']):
            if not Path(name).is_file():
                raise FileNotFoundError(name)
        routes = entry.get('routes', ())
        if policy == 'candidate_graph' and 'routes' in entry:
            if [row['capture'] for row in routes] != paths:
                raise ValueError('manifest capture roles disagree with route records')
        for row in routes:
            capture = row['capture']
            if policy == 'signal_graph' and capture not in paths:
                raise ValueError('route is outside declared capture roles')
            actual = hashlib.sha256(Path(capture).read_bytes()).hexdigest()
            if actual != row['converted_sha256']:
                reason = ('converted capture hash changed' if policy == 'candidate_graph'
                          else 'capture hash differs from manifest')
                raise ValueError(f'{reason}: {capture}')
            route_hashes.append(row['converted_sha256'])
            if policy == 'candidate_graph':
                if row.get('vehicle') != entry['vehicle'] or row.get('bus') != 0:
                    raise ValueError('manifest vehicle or bus mismatch')
                if not row.get('dataset_revision') or not row.get('opendbc_revision'):
                    raise ValueError('manifest lacks source revisions')
                if len(row.get('compressed_sha256', '')) != 64:
                    raise ValueError('manifest lacks compressed segment hash')
                if hashlib.sha256(Path(entry['dbc']).read_bytes()).hexdigest() != row['dbc_sha256']:
                    raise ValueError('DBC hash changed')
        if routes:
            if len({row['device'] for row in routes}) != len(routes):
                raise ValueError('routes must have distinct devices')
            if len({row['session'] for row in routes}) != len(routes):
                raise ValueError('routes must have distinct sessions')
    if len(all_paths) != len(set(all_paths)):
        raise ValueError('source and target capture paths overlap')
    if len(route_hashes) != len(set(route_hashes)):
        raise ValueError('source and target capture hashes overlap')
    return entries
