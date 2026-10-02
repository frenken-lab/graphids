"""Fixed-geometry, same-section pipeline diagnostics in fresh processes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphids.exp.config import load_experiment_yaml
from graphids.exp.trials import run_trial


def run_trials(root, *, fusion=False):
    root.mkdir(parents=True, exist_ok=True)
    base = load_experiment_yaml('configs/experiments/comparisons/temporal_set_01_segmented_lanes256_scaling_diagnostic.yml')
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
        result = run_trial(root, name, config=config, section_limit=1024, reference=next(iter(results.values()), None))
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
