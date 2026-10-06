"""Run the held-out physical benchmark for each pinned source vehicle."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark_manifest', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-ids', type=int, default=64)
    args = parser.parse_args()
    script = Path(__file__).with_name('experiment_can_physical.py')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in json.loads(args.benchmark_manifest.read_text()):
        output = args.output_dir / f"2026-10-04-{entry['vehicle'].lower()}-physical-v6.json"
        command = [sys.executable, str(script), *entry['references'], entry['held_out'],
                   '--target-vehicle', entry['vehicle'], '--evaluation-dbc', entry['dbc'],
                   '--audit-candidate-pool', '--max-ids', str(args.max_ids),
                   '--output', str(output)]
        if args.source_manifest:
            command += ['--source-manifest', str(args.source_manifest)]
            command += ['--source-benchmark-manifest', str(args.benchmark_manifest)]
        subprocess.run(command, check=True)
        evaluation = json.loads(output.read_text())['evaluation']
        print(json.dumps({'vehicle': entry['vehicle'], 'output': str(output),
                          'selected_layout': evaluation['selected_layout'],
                          'candidate_pool': evaluation['candidate_pool'],
                          'roles': {role: {
                              'exact_rank': item['first_exact_rank'],
                              'value_equivalent_rank': item['first_value_equivalent_rank']}
                              for role, item in evaluation['roles'].items()}},
                         separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
