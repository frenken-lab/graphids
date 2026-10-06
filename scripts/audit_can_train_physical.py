"""Audit provisional physical-role fields at the paper's CAN message anchors."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from audit_can_train_signal_anchors import ANCHORS, PAPER

from can_discovery.preprocessing.capture import load_capture
from can_discovery.relationships.physical import (
    fit_physical_roles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vehicle', choices=ANCHORS)
    parser.add_argument('reference_a', type=Path)
    parser.add_argument('reference_b', type=Path)
    parser.add_argument('--max-ids', type=int, default=32)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    captures = [load_capture(path) for path in (args.reference_a, args.reference_b)]
    if captures[0].sha256 == captures[1].sha256:
        raise SystemExit('normal recordings are exact duplicates')
    if any(any(frame.attack for frame in capture.frames) for capture in captures):
        raise SystemExit('anchor audit requires attack-free normal files')
    result = fit_physical_roles(captures, max_ids=args.max_ids,
                                message_ids=tuple(ANCHORS[args.vehicle]))
    anchors = {}
    for arb_id, paper_role in ANCHORS[args.vehicle].items():
        role = {'displayed_speed': 'speed'}.get(paper_role, paper_role)
        candidates = [asdict(item) for item in result.hypotheses
                      if item.role == role and item.candidate.arb_id.upper().zfill(3) == arb_id]
        anchors[arb_id] = {'paper_message_role': paper_role,
                           'candidate_fields': candidates,
                           'field_label_validated': False}
    report = {'version': 6, 'vehicle': args.vehicle, 'paper': PAPER,
              'references': [{'path': str(path), 'sha256': capture.sha256}
                             for path, capture in zip((args.reference_a, args.reference_b), captures)],
              'held_out_normal_trip_available': False,
              'evidence_level': 'message_anchor_plus_provisional_field_hypotheses',
              'anchors': anchors,
              'combinations': [asdict(item) for item in result.combinations],
              'rejected': result.rejected}
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
