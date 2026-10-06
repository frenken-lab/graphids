"""Compare inferred CAN layouts with a known paired DBC."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from can_discovery.decoding.dbc import (
    evaluate_discovery,
    load_dbc_fields,
    rank_dbc,
)
from can_discovery.decoding.signals import discover_fields
from can_discovery.preprocessing.capture import load_capture


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_a", type=Path)
    parser.add_argument("reference_b", type=Path)
    parser.add_argument("held_out", type=Path)
    parser.add_argument("dbc", type=Path)
    parser.add_argument("--max-ids", type=int, default=64)
    parser.add_argument("--samples-per-id", type=int, default=512)
    parser.add_argument("--min-changes", type=int, default=20)
    parser.add_argument("--tokenizer", choices=('can_d', 'bit_flip', 'bycan'), default='can_d')
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    references = [load_capture(args.reference_a), load_capture(args.reference_b)]
    held_out = load_capture(args.held_out)
    fits = [rank_dbc(capture, args.dbc) for capture in (*references, held_out)]
    if any(fit["coverage"] < .5 or
           fit["matched_length_frames"] != fit["matched_id_frames"] for fit in fits):
        raise SystemExit("vehicle/bus/length does not fit the supplied DBC on all routes")
    start = time.perf_counter()
    registry = discover_fields(references, max_ids=args.max_ids,
                               samples_per_id=args.samples_per_id,
                               min_changes=args.min_changes, max_fields_per_id=16,
                               method=args.tokenizer)
    fields = registry.selected
    elapsed = time.perf_counter() - start
    truth = load_dbc_fields(args.dbc)
    result = {"method": args.tokenizer, "seconds": elapsed,
              "references": [str(args.reference_a), str(args.reference_b)],
              "held_out": str(args.held_out), "dbc": str(args.dbc),
              "dbc_fit": fits[-1], "reference_dbc_fit": fits[:2],
              "evaluation": evaluate_discovery(fields, truth, held_out),
              "tokenizations": [{"id": item.trace.arb_id,
                                  "payload_length": item.trace.payload_length,
                                  "bus": item.trace.bus,
                                  "parameters": item.parameters,
                                  "source_hashes": item.trace.source_hashes,
                                  "support": item.trace.support,
                                  "spans": [{"physical_bits": span.physical_bits,
                                             "byte_order": span.byte_order}
                                            for span in item.spans],
                                  "uncovered_bits": item.uncovered_bits,
                                  "abstention_reasons": item.abstention_reasons}
                                 for item in registry.tokenizations],
              "protocol_conflicts": registry.conflicts,
              "fields": [{"id": f.arb_id, "payload_length": f.payload_length,
                          "start": f.start, "width": f.width,
                          "endian": f.endian, "signed": f.signed,
                          "role": f.role, "confidence": f.confidence,
                          "evidence": f.evidence,
                          "mux_start": f.mux_start, "mux_value": f.mux_value}
                         for f in fields]}
    content = json.dumps(_json_safe(result), indent=2, default=str, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n")
    else:
        print(content)


if __name__ == "__main__":
    main()
