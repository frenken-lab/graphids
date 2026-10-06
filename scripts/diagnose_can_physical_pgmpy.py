"""Optional time-sliced pgmpy diagnostic for an already selected role graph."""
from __future__ import annotations

import argparse
import json
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from can_discovery.decoding.signals import FieldCandidate
from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.shared.signal_time import align
from can_discovery.relationships.physical.role_types import ROLES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('discovery_report', type=Path)
    parser.add_argument('reference_a', type=Path)
    parser.add_argument('reference_b', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.loads(args.discovery_report.read_text())
    best = report['combinations'][0]
    roles = [role for role, index in zip(ROLES, best['assignments']) if index is not None]
    fields = [FieldCandidate(**report['hypotheses'][index]['candidate'])
              for index in best['assignments'] if index is not None]
    result = {'status': 'unavailable', 'roles': roles,
              'interpretation': 'time-sliced conditional-dependence diagnostic; no causal field labels'}
    if find_spec('pgmpy') is None:
        result['reason'] = 'pgmpy_not_installed'
    elif len(fields) < 2:
        result['reason'] = 'too_few_roles'
    else:
        import pandas as pd
        from pgmpy.causal_discovery import PC, ExpertKnowledge

        rows = []
        for path in (args.reference_a, args.reference_b):
            values, valid, _, _ = align(load_capture(path), tuple(fields), step=.1, max_age=.3)
            mask = np.all(valid[1:] & valid[:-1], axis=1)
            if mask.sum() < 100:
                continue
            current = values[:-1][mask]
            future = values[1:][mask]
            data = np.column_stack((current, future))
            scale = np.maximum(np.std(data, axis=0), 1e-8)
            rows.append(data / scale)
        if rows:
            columns = [f'{role}_t' for role in roles] + [f'{role}_next' for role in roles]
            frame = pd.DataFrame(np.concatenate(rows), columns=columns)
            expert = ExpertKnowledge(temporal_order=[columns[:len(roles)], columns[len(roles):]])
            try:
                model = PC(ci_test='pearsonr', return_type='pdag', max_cond_vars=2,
                           expert_knowledge=expert, enforce_expert_knowledge=True,
                           show_progress=False)
                model.fit(frame)
                result.update(status='completed', edges=[list(edge) for edge in
                              model.causal_graph_.edges()], sample_count=len(frame))
            except (TypeError, ValueError, AttributeError) as error:
                result['reason'] = f'pgmpy_diagnostic_failed: {type(error).__name__}: {error}'
        else:
            result['reason'] = 'insufficient_joint_samples'
    content = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
