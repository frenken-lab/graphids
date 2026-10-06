"""Stage 0 audit and structural reports for CAN-Train-and-Test normal captures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from can_discovery.preprocessing.capture import audit_captures, load_capture
from can_discovery.preprocessing.tokenization import prepare_traces, tokenize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-ids', type=int, default=64)
    parser.add_argument('--frames-per-capture', type=int, default=1000)
    args = parser.parse_args()
    report = {}
    for folder in sorted(args.raw_root.glob('set_*')):
        paths = sorted((folder/'train_01_attack_free').glob('*.csv'))
        if len(paths) < 2:
            continue
        captures = [load_capture(path) for path in paths]
        audit = audit_captures(captures)
        traces = prepare_traces(captures,max_ids=args.max_ids,
                                samples_per_capture=args.frames_per_capture)
        methods = {}
        for method in ('bit_flip','can_d','bycan'):
            results = [tokenize(trace,method) for trace in traces]
            methods[method] = [{'id':r.trace.arb_id,'length':r.trace.payload_length,
                'bus':r.trace.bus,'support':r.trace.support,
                'source_hashes':r.trace.source_hashes,'parameters':r.parameters,
                'spans':[{'physical_bits':s.physical_bits,'byte_order':s.byte_order}
                         for s in r.spans], 'uncovered_bits':r.uncovered_bits,
                'abstention_reasons':r.abstention_reasons} for r in results]
        report[folder.name] = {'audit':audit,'eligible_families':len(traces),
                               'methods':methods,'field_accuracy':None}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'version':1,'sets':report},indent=2)+'\n')


if __name__ == '__main__':
    main()
