"""Check physical-time relationships independently in source vehicle trips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from can_discovery.decoding.dbc import load_dbc_fields, rank_dbc
from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical import (
    _dbc_role,
    _index_capture,
    _relation_score,
    _sample_values,
)

RELATIONS = {'speed_rpm': ('speed', 'rpm'),
             'speed_acceleration': ('speed', 'acceleration'),
             'speed_brake': ('speed', 'brake'),
             'rpm_throttle': ('rpm', 'throttle')}


def _select(fields, role, captures, indices):
    ranked = []
    for field in fields:
        if _dbc_role(field.label_hypotheses[0][0]) != role:
            continue
        traces = [_sample_values(capture, field, cap=2048, index=index)[1]
                  for capture, index in zip(captures, indices)]
        if min(map(len, traces), default=0) < 30 or min(len(np.unique(x)) for x in traces) < 2:
            continue
        ranked.append((min(len(np.unique(x)) for x in traces), field))
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = []
    for entry in json.loads(args.source_manifest.read_text()):
        captures = [load_capture(Path(path)) for path in entry['captures']]
        dbc = Path(entry['dbc'])
        fits = [rank_dbc(capture, dbc) for capture in captures]
        if any(fit['coverage'] < .5 for fit in fits):
            raise SystemExit(f"source DBC preflight failed: {entry['vehicle']}")
        fields = load_dbc_fields(dbc)
        indices = [_index_capture(capture) for capture in captures]
        chosen = {role: _select(fields, role, captures, indices)
                  for role in {role for pair in RELATIONS.values() for role in pair}}
        relations = {}
        for name, (left, right) in RELATIONS.items():
            a, b = chosen[left], chosen[right]
            if a is None or b is None:
                relations[name] = {'status': 'unavailable_role',
                                   'missing': [role for role, field in ((left, a), (right, b))
                                               if field is None]}
                continue
            effects = [_relation_score([capture], a, b, name, [index])
                       for capture, index in zip(captures, indices)]
            relations[name] = {'status': 'retained' if min(effects) > .05 else 'failed',
                               'fields': [a.label_hypotheses[0][0], b.label_hypotheses[0][0]],
                               'fit_effect': effects[0], 'held_out_effect': effects[1]}
        report.append({'vehicle': entry['vehicle'], 'capture_hashes': [c.sha256 for c in captures],
                       'dbc_fit': fits, 'relations': relations})
    transfer = {name: {'supported_vehicles': [item['vehicle'] for item in report
                                               if item['relations'][name]['status'] == 'retained'],
                       'unavailable_vehicles': [item['vehicle'] for item in report
                                                if item['relations'][name]['status'] == 'unavailable_role']}
                for name in RELATIONS}
    for name, entry in transfer.items():
        entry['cross_vehicle_retained'] = len(entry['supported_vehicles']) >= 2
    result = {'vehicles': report, 'transfer': transfer,
              'method': '0.1s_masked_alignment_shift_control_two_distinct_trips_per_vehicle'}
    content = json.dumps(result, indent=2, allow_nan=False, default=str) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
