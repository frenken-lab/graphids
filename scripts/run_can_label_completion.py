"""Run the current DBC label-completion development workflow.

Prepare every case in a track before ranking. DBC-assisted source gates read
prepared cases from the other vehicles. Final routes remain sealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from can_discovery.label_completion.rank import run as rank_case
from can_discovery.label_completion.case import TRACKS, run as prepare_case
from can_discovery.manifests.label_completion import write_artifact
from can_discovery.label_completion.report import aggregate, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'rank', 'report'))
    parser.add_argument('staged_manifest', type=Path)
    parser.add_argument('source_manifest', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--track', choices=TRACKS, required=True)
    parser.add_argument('--case-index', type=int)
    args = parser.parse_args()
    if args.phase in ('prepare', 'rank') and args.case_index is None:
        parser.error('--case-index is required for prepare and rank')
    if args.phase == 'report' and args.case_index is not None:
        parser.error('--case-index is not used for report')
    track_dir = args.output_dir / args.track
    if args.phase == 'prepare':
        result = prepare_case(args.staged_manifest, track_dir,
                              args.case_index, args.source_manifest, args.track)
    elif args.phase == 'rank':
        result = rank_case(args.staged_manifest, args.source_manifest,
                           track_dir, track_dir, track_dir,
                           args.case_index, args.track)
    else:
        result = aggregate(args.staged_manifest, args.source_manifest,
                           track_dir, args.track)
        write_artifact(track_dir / 'summary.json', result)
        content = report(result) + '\n'
        destination = track_dir / 'REPORT.md'
        if destination.exists() and destination.read_text() != content:
            raise ValueError(f'immutable report changed: {destination}')
        destination.write_text(content)
    print(json.dumps(result['metrics'] if args.phase == 'report' else {
        'vehicle': result['vehicle'],
        'role': result.get('role', result.get('hidden_role')),
        'phase': args.phase,
        'track': args.track,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
