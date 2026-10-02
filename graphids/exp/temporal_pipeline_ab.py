"""Fixed-geometry, same-section pipeline diagnostics in fresh processes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from graphids.exp.temporal_lane_sweep import validate_trial


def run_trials(root, *, fusion=False):
    root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path('configs/experiments/comparisons/temporal_set_01_segmented_lanes256_scaling_diagnostic.yml').read_text())
    variants = [('workers0', 0), ('workers2', 2), ('workers4', 4), ('workers0_repeat', 0)]
    if fusion:
        variants = [('torch', 0), ('fused', 0), ('torch_repeat', 0), ('fused_repeat', 0)]
    results = {}
    for name, workers in variants:
        config = json.loads(json.dumps(base))
        config['experiment_name'] = 'temporal_pipeline_' + name
        data = config['config']['data']
        data.update(num_workers=workers, prebuild_batches=2 if workers else 0,
                    persistent_workers=bool(workers), pin_memory=False)
        if fusion and name.startswith('fused'):
            config['config']['model']['scan']['ssm_kernel'] = 'triton'
        path = root / (name + '.yml')
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        out = root / name
        subprocess.run([sys.executable, '-m', 'graphids.exp.temporal_comparison', str(path),
                        '--output', str(out), '--section-limit', '1024'], check=True)
        result = json.loads((out / 'comparison.json').read_text())
        validate_trial(result, next(iter(results.values()), None))
        results[name] = result
        (root / 'pipeline.json').write_text(json.dumps(results, indent=2) + '\n')
        print('pipeline_trial ' + json.dumps({'name': name, 'rate': result['coverage']['scored_events_per_second']}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--fusion', action='store_true')
    args = parser.parse_args()
    run_trials(args.root, fusion=args.fusion)


if __name__ == '__main__':
    main()
