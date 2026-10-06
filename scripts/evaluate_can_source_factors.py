"""Audit each physics factor on two source trips and an independent third trip."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from can_discovery.decoding.dbc import load_dbc_fields, rank_dbc
from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical import (
    _dbc_role,
    _sample_values,
    fit_source_factor_validation,
)
from can_discovery.relationships.physical.physics_factors import (
    FACTOR_ROLES,
    evaluate_factor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark_manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    entries = json.loads(args.benchmark_manifest.read_text())
    sources = [(entry['vehicle'],
                [load_capture(Path(path)) for path in (*entry['references'], entry['held_out'])],
                Path(entry['dbc'])) for entry in entries]
    vehicles = []
    for vehicle, trips, dbc in sources:
        fits = [rank_dbc(trip, dbc) for trip in trips]
        if any(item['coverage'] < .5 for item in fits):
            raise SystemExit(f'source DBC preflight failed: {vehicle}')
        chosen = {}
        for field in load_dbc_fields(dbc):
            role = _dbc_role(field.label_hypotheses[0][0])
            if role is None:
                continue
            activity = min(len(np.unique(_sample_values(trip, field, cap=512)[1]))
                           for trip in trips[:2])
            if activity > chosen.get(role, (0, None))[0]:
                chosen[role] = (activity, field)
        factors = {}
        for name, roles in FACTOR_ROLES.items():
            if any(role not in chosen for role in roles):
                factors[name] = {'status': 'unavailable', 'reason': 'missing_role'}
                continue
            fields = tuple(chosen[role][1] for role in roles)
            fitted = evaluate_factor(trips[:2], fields, name)
            held = evaluate_factor(trips[2:], fields, name, forced_lag=fitted.lag_bins)
            factors[name] = {'status': 'retained' if fitted.status == held.status == 'supported'
                              and fitted.direction == held.direction else 'failed',
                             'fields': [field.label_hypotheses[0][0] for field in fields],
                             'fit': asdict(fitted), 'held_out': asdict(held)}
        vehicles.append({'vehicle': vehicle, 'capture_hashes': [trip.sha256 for trip in trips],
                         'dbc_fit': fits, 'factors': factors})
    validated = fit_source_factor_validation(sources, target_vehicle='__none__')
    report = {'version': 6, 'vehicles': vehicles,
              'validated_transfer_factors': [asdict(item) for item in validated],
              'interpretation': 'predictive source support, not causal proof or target field truth'}
    content = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
