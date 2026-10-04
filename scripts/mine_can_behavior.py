"""Fit and score all CAN behavior discovery families on frozen normal roles."""
from __future__ import annotations

import argparse
from pathlib import Path

from graphids.core.data.discovery.behavior.runner import run_set
from graphids.paths import lake_root

SETS = ('set_01', 'set_02', 'set_03', 'set_04', 'hcrl_ch', 'hcrl_sa')
DEFAULT_SETS = SETS[:4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--pilot-root', type=Path, help='frozen assignment root; defaults to deterministic assignment')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--normal-only', action='store_true')
    parser.add_argument('--sets', nargs='+', choices=SETS, default=DEFAULT_SETS)
    args = parser.parse_args()
    raw_root = args.raw_root or Path(lake_root()) / 'raw'
    for name in args.sets:
        report = run_set(raw_root / name, args.output / name,
                         frozen=args.pilot_root / name if args.pilot_root else None,
                         include_attacks=not args.normal_only)
        print(name, report['rule_counts'], flush=True)


if __name__ == '__main__':
    main()
