"""Run the current DBC label-completion development workflow.

Prepare every case before ranking: source-side contrast gates read prepared
cases from the other vehicles. Final routes remain sealed in all three phases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from can_discovery.label_completion.current import run as rank_case
from can_discovery.label_completion.full_search import run as prepare_case
from can_discovery.label_completion.manifest import write_artifact
from can_discovery.label_completion.report import aggregate, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'rank', 'report'))
    parser.add_argument('staged_manifest', type=Path)
    parser.add_argument('source_manifest', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--case-index', type=int)
    args = parser.parse_args()
    if args.phase in ('prepare', 'rank') and args.case_index is None:
        parser.error('--case-index is required for prepare and rank')
    if args.phase == 'report' and args.case_index is not None:
        parser.error('--case-index is not used for report')
    if args.phase == 'prepare':
        result = prepare_case(args.staged_manifest, args.output_dir,
                              args.case_index, args.source_manifest)
    elif args.phase == 'rank':
        result = rank_case(args.staged_manifest, args.source_manifest,
                           args.output_dir, args.output_dir, args.output_dir,
                           args.case_index)
    else:
        result = aggregate(args.staged_manifest, args.source_manifest,
                           args.output_dir)
        write_artifact(args.output_dir / 'summary.json', result)
        content = report(result) + '\n'
        destination = args.output_dir / 'REPORT.md'
        if destination.exists() and destination.read_text() != content:
            raise ValueError(f'immutable report changed: {destination}')
        destination.write_text(content)
    print(json.dumps(result['metrics'] if args.phase == 'report' else {
        'vehicle': result['vehicle'],
        'role': result.get('role', result.get('hidden_role')),
        'phase': args.phase,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
