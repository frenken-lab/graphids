"""Audit and compare tokenizers on pinned independent captures."""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

from can_discovery.decoding.dbc import load_dbc_fields
from can_discovery.decoding.signals import _bit_positions
from can_discovery.preprocessing.capture import audit_captures, load_capture
from can_discovery.preprocessing.tokenization import prepare_traces, tokenize
from can_discovery.preprocessing.source_model import train_source_model


def _candidate_recall(results, truth, held_out):
    from can_discovery.decoding.field_values import extract
    held = {}
    for frame in held_out.frames:
        if frame.payload is not None:
            held.setdefault((frame.arb_id.upper(),len(frame.payload)),[]).append(frame.payload)
    actual = set()
    for field in truth:
        key = (field.arb_id.upper(),field.payload_length)
        if field.mux_start is None and len({extract(payload,field.start,field.width,
                field.endian,field.signed) for payload in held.get(key,())}) > 1:
            actual.add((*key,_bit_positions(field.start,field.width,field.endian)))
    proposed = {(*((r.trace.arb_id.upper(),r.trace.payload_length)),frozenset(s.physical_bits))
                for r in results for layout in r.alternatives for s in layout.spans}
    return len(actual & proposed)/max(1,len(actual))


def _metrics(results, truth, held_out):
    families = {(r.trace.arb_id.upper(), r.trace.payload_length) for r in results}
    known = []
    if truth is not None:
        held = {}
        for frame in held_out.frames:
            if frame.payload is not None:
                held.setdefault((frame.arb_id.upper(), len(frame.payload)), []).append(frame.payload)
        for field in truth:
            key = (field.arb_id.upper(), field.payload_length)
            if key not in families or field.mux_start is not None:
                continue
            from can_discovery.decoding.field_values import extract
            values = held.get(key, ())
            if len({extract(p, field.start, field.width, field.endian, field.signed)
                    for p in values}) > 1:
                known.append(field)
    predicted = {(r.trace.arb_id.upper(), r.trace.payload_length, frozenset(s.physical_bits))
                 for r in results for s in r.spans}
    actual = {(f.arb_id.upper(), f.payload_length,
               _bit_positions(f.start, f.width, f.endian)) for f in known}
    # Score only physically adjacent, documented active bits. DBC start/width
    # coordinates do not themselves identify a physical boundary.
    truth_owner = {}
    for index,f in enumerate(known):
        for bit in _bit_positions(f.start,f.width,f.endian):
            truth_owner.setdefault((f.arb_id.upper(),f.payload_length,bit),set()).add(index)
    predicted_owner = {}
    for result in results:
        for index,span in enumerate(result.spans):
            for bit in span.physical_bits:
                predicted_owner[(result.trace.arb_id.upper(),result.trace.payload_length,bit)] = index
    pb,ab = set(),set()
    for result in results:
        family = (result.trace.arb_id.upper(),result.trace.payload_length)
        for bit in range(result.trace.payload_length*8-1):
            if bit//8 != (bit+1)//8:
                continue
            left,right = family+(bit,),family+(bit+1,)
            if left not in truth_owner or right not in truth_owner:
                continue
            key = family+(bit,)
            if truth_owner[left].isdisjoint(truth_owner[right]):
                ab.add(key)
            if predicted_owner.get(left) != predicted_owner.get(right):
                pb.add(key)
    resolved = {(r.trace.arb_id.upper(), r.trace.payload_length,
                 frozenset(s.physical_bits)): s.byte_order
                for r in results for s in r.spans if s.byte_order != 'unresolved'}
    cross_byte = [f for f in known if len({bit//8 for bit in
                  _bit_positions(f.start,f.width,f.endian)}) > 1]
    correct_order = sum(resolved.get((f.arb_id.upper(),f.payload_length,
                         _bit_positions(f.start,f.width,f.endian))) == f.endian for f in cross_byte)
    order_total = sum((f.arb_id.upper(),f.payload_length,
                       _bit_positions(f.start,f.width,f.endian)) in resolved for f in cross_byte)
    return {'boundary_precision': len(pb & ab)/max(1,len(pb)) if truth else None,
            'boundary_recall': len(pb & ab)/max(1,len(ab)) if truth else None,
            'exact_layout_precision': len(predicted & actual)/max(1,len(predicted)) if truth else None,
            'exact_layout_recall': len(predicted & actual)/max(1,len(actual)) if truth else None,
            'byte_order_accuracy': correct_order/order_total if order_total else None,
            'byte_order_assessed': order_total,
            'abstentions': sum(bool(r.abstention_reasons) for r in results),
            'unresolved_order_spans': sum(s.byte_order == 'unresolved' for r in results for s in r.spans),
            'frame_coverage': sum(1 for f in held_out.frames if (f.arb_id.upper(),len(f.payload or b'')) in families)/max(1,len(held_out.frames)),
            'dbc_family_coverage': len(families & {(f.arb_id.upper(),f.payload_length) for f in truth})/max(1,len({(f.arb_id.upper(),f.payload_length) for f in truth})) if truth else None,
            'per_family_failures': [{'id': a, 'length': n} for a,n,b in sorted(actual-predicted)] if truth else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-ids', type=int, default=64)
    parser.add_argument('--supervised', action='store_true')
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    entries = entries['captures'] if isinstance(entries, dict) else entries
    report = []
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else args.manifest.parent / path
    for entry in entries:
        references = [load_capture(resolve(p), metadata=entry) for p in entry['references']]
        held_out = load_capture(resolve(entry['held_out']), metadata=entry)
        truth = load_dbc_fields(resolve(entry['dbc'])) if entry.get('dbc') else None
        model = None
        if args.supervised:
            sources = []
            for source in entries:
                if source['vehicle'] == entry['vehicle']:
                    continue
                captures = [load_capture(resolve(p), metadata=source)
                            for p in [*source['references'],source['held_out']]]
                sources.append((source['vehicle'],captures,resolve(source['dbc'])))
            model = train_source_model(sources)
            model.check_target([c.sha256 for c in [*references,held_out]], entry['vehicle'])
        audits = audit_captures([*references,held_out])
        budgets = [1000,10000]
        runs = []
        for budget in budgets:
            if min(sum(f.payload is not None for f in c.frames) for c in references) < budget:
                if budget == 10000:
                    continue
            traces = prepare_traces(references,max_ids=args.max_ids,samples_per_capture=budget)
            methods = ('bit_flip','can_d','bycan') + (('supervised_can_d',) if model else ())
            for method in methods:
                layout_counts = (1,8) if method == 'supervised_can_d' else (1,)
                for layout_count in layout_counts:
                    tracemalloc.start()
                    started = time.perf_counter()
                    results = tuple(tokenize(trace,method,source_model=model if method == 'supervised_can_d' else None,
                                             max_layouts=layout_count) for trace in traces)
                    seconds = time.perf_counter()-started
                    _,peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    runs.append({'method':method,'max_layouts':layout_count,'frames_per_reference':budget,
                                 'fit_seconds':seconds,'peak_python_bytes':peak,
                                 'eligible_families':len(traces),
                                 'metrics':_metrics(results,truth,held_out),
                                 'candidate_recall': None if truth is None else _candidate_recall(results,truth,held_out),
                                 'tokenizations':[{'id':r.trace.arb_id,'length':r.trace.payload_length,
                                     'bus':r.trace.bus,'source_hashes':r.trace.source_hashes,
                                     'support':r.trace.support,'parameters':r.parameters,
                                     'spans':[{'physical_bits':s.physical_bits,'byte_order':s.byte_order}
                                              for s in r.spans],
                                     'alternative_costs':[a.cost for a in r.alternatives],
                                     'uncovered_bits':r.uncovered_bits,
                                     'abstention_reasons':r.abstention_reasons} for r in results]})
        report.append({'vehicle':entry.get('vehicle'),'audit':audits,'runs':runs,
                       'truth_available':truth is not None})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'version':1,'entries':report},indent=2)+'\n')


if __name__ == '__main__':
    main()
