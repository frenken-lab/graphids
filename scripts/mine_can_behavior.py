"""Fit and score all CAN behavior discovery families on frozen normal roles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from can_discovery.behavior.runner import run_set
from can_discovery.paths import raw_root
from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical import (
    fit_source_calibration,
    fit_source_factor_validation,
)
from can_discovery.vehicles import declared_vehicle_id

SETS = ('set_01', 'set_02', 'set_03', 'set_04', 'hcrl_ch', 'hcrl_sa')
DEFAULT_SETS = SETS[:4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--pilot-root', type=Path, help='frozen assignment root; defaults to deterministic assignment')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sets', nargs='+', choices=SETS, default=DEFAULT_SETS)
    parser.add_argument('--source-manifest', type=Path,
                        help='JSON source-vehicle normal captures and DBCs for role calibration')
    parser.add_argument('--source-benchmark-manifest', type=Path,
                        help='three-trip DBC source benchmark for validated factor weights')
    args = parser.parse_args()
    data_root = args.raw_root or raw_root()
    sources = None
    if args.source_manifest:
        entries = json.loads(args.source_manifest.read_text())
        sources = [(entry['vehicle'], [load_capture(Path(path)) for path in entry['captures']],
                    Path(entry['dbc'])) for entry in entries]
    factor_sources = None
    if args.source_benchmark_manifest:
        entries = json.loads(args.source_benchmark_manifest.read_text())
        factor_sources = [(entry['vehicle'],
                           [load_capture(Path(path)) for path in (*entry['references'], entry['held_out'])],
                           Path(entry['dbc'])) for entry in entries]
    for name in args.sets:
        calibration = (fit_source_calibration(
            sources, target_vehicle=declared_vehicle_id(name, 'train_01_attack_free'))
            if sources is not None else ())
        validated = (fit_source_factor_validation(
            factor_sources, target_vehicle=declared_vehicle_id(name, 'train_01_attack_free'))
            if factor_sources is not None else ())
        report = run_set(data_root / name, args.output / name,
                         frozen=args.pilot_root / name if args.pilot_root else None,
                         source_calibration=calibration, validated_factors=validated)
        print(name, report['rule_counts'], flush=True)


if __name__ == '__main__':
    main()
