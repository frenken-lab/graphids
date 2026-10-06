"""Fit and assess the experimental modular vehicle graph without a target DBC.

Run with ``python -m scripts.benchmark_can_combined_graph MANIFEST OUTPUT``.
The manifest has fitting/development/final capture paths and optionally frozen
route hashes. The final route is read only with --sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical import ROLES, fit_physical_roles
from can_discovery.relationships.physical.combined_graph import evaluate_combined_graph


def _paths(entry, role):
    if role == 'fitting':
        return entry.get('fitting', entry.get('references', ()))
    if role == 'development':
        return entry.get('development', entry.get('validation', ()))
    return [entry['final']] if entry.get('final') else ()


def _load(entry, role):
    paths = _paths(entry, role)
    if not paths:
        return []
    expected = {str(route['capture']): route['converted_sha256']
                for route in entry.get('routes', ())}
    captures = []
    for path_text in paths:
        path = Path(path_text)
        if expected and str(path) not in expected:
            raise ValueError(f'{role} capture absent from frozen manifest: {path}')
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected[str(path)]:
            raise ValueError(f'frozen capture hash changed: {path}')
        captures.append(load_capture(path))
    return captures


def run_entry(entry: dict, *, sealed: bool = False, max_ids: int = 64) -> dict:
    started = time.perf_counter()
    fits = _load(entry, 'fitting')
    if len(fits) < 2 or len({item.sha256 for item in fits}) != len(fits):
        raise ValueError('two distinct fitting routes are required')
    # This function receives no DBC argument; fitting and field selection use
    # only CAN payloads. Preserve the ranked alternatives in the artifact.
    discovery = fit_physical_roles(fits, max_ids=max_ids)
    if not discovery.combinations:
        raise ValueError('physical fit returned no candidate combinations')
    selection = discovery.combinations[0]
    fields = {role: discovery.hypotheses[index].candidate
              for role, index in zip(ROLES, selection.assignments, strict=True)
              if index is not None}
    fitted = evaluate_combined_graph(fits, fields,
                                     powertrain=entry.get('powertrain', 'unknown'))
    result = {'vehicle': entry['vehicle'], 'version': 1,
              'selected_fields': {role: asdict(field) for role, field in fields.items()},
              'combination_weight': selection.weight,
              'combination_score': selection.score,
              'ranked_combination_count': len(discovery.combinations),
              'fit': fitted, 'fit_seconds': time.perf_counter()-started,
              'promotion_status': 'experimental_unpromoted'}
    seen = set(fitted['capture_hashes'])
    for role in ('development', 'final'):
        if role == 'final' and not sealed:
            continue
        captures = _load(entry, role)
        if not captures:
            continue
        hashes = {capture.sha256 for capture in captures}
        if len(hashes) != len(captures) or seen & hashes:
            raise ValueError('capture overlap between fit and assessment')
        seen |= hashes
        result[role] = evaluate_combined_graph(captures, fields,
                                               frozen=fitted['fit_choices'],
                                               powertrain=entry.get('powertrain', 'unknown'))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sealed', action='store_true',
                        help='assess the declared final captures once after fit and development')
    parser.add_argument('--max-ids', type=int, default=64)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    if not isinstance(entries, list):
        raise ValueError('expected a list of vehicle entries')
    results = [run_entry(entry, sealed=args.sealed, max_ids=args.max_ids)
               for entry in entries]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
