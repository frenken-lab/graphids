"""Audit paper message anchors at field level without assigning physical names.

Fit on the first half of each normal driving recording; inspect the withheld
second halves for candidate variation. These halves are not independent trips.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from audit_can_train_signal_anchors import ANCHORS

from can_discovery.decoding.field_values import extract
from can_discovery.decoding.signals import fit_fields
from can_discovery.preprocessing.capture import Capture, load_capture
from can_discovery.relationships.shared.signal_time import align


def half(capture: Capture, start: int, stop: int) -> Capture:
    frames = capture.frames[start:stop]
    return Capture(capture.source, capture.sha256 + f":{start}:{stop}", frames,
                   frames[0].timestamp, frames[-1].timestamp, 0, 0, {}, {})


def audit(vehicle: str, paths: list[Path]) -> dict:
    anchors = ANCHORS[vehicle]
    recordings = []
    for path in paths:
        capture = load_capture(path)
        frames = tuple(frame for frame in capture.frames
                       if frame.arb_id.upper().zfill(3) in anchors)
        recordings.append(Capture(capture.source, capture.sha256, frames,
                                  frames[0].timestamp, frames[-1].timestamp,
                                  0, 0, {}, {}))
    refs = [half(capture, 0, len(capture.frames) // 2) for capture in recordings]
    held = [half(capture, len(capture.frames) // 2, len(capture.frames))
            for capture in recordings]
    fields = fit_fields(refs, max_ids=8, samples_per_id=512,
                        max_fields_per_id=16)
    result = {"vehicle": vehicle, "evidence_level": "field_variation_only",
              "reference_paths": [str(path) for path in paths],
              "split": "first_half_fit_second_half_check_same_trips",
              "anchors": {}, "validated_physical_labels": [],
              "unavailable_roles": sorted(set(anchors.values()))}
    for identifier, role in anchors.items():
        candidates = []
        for field in fields:
            if field.arb_id.upper().zfill(3) != identifier:
                continue
            checks = []
            for capture in held:
                values = [extract(frame.payload, field.start, field.width,
                                  field.endian, field.signed)
                          for frame in capture.frames if frame.arb_id.upper().zfill(3) == identifier
                          and frame.payload is not None and len(frame.payload) == field.payload_length]
                values = np.asarray([value for value in values if value is not None], dtype=float)
                checks.append({"observations": len(values),
                               "distinct_values": len(np.unique(values)),
                               "change_fraction": (float(np.mean(np.diff(values) != 0))
                                                   if len(values) > 1 else None)})
            candidates.append({"start": field.start, "width": field.width,
                               "endian": field.endian, "signed": field.signed,
                               "structural_role": field.role, "held_out": checks})
        result["anchors"][identifier] = {"paper_message_role": role,
                                          "candidate_fields": candidates,
                                          "physical_label_status": "unresolved"}
    checks = []
    for left_id, right_id in (("3E9", "0C9"), ("3E9", "1F5")):
        left = [field for field in fields if field.arb_id.upper().zfill(3) == left_id]
        right = [field for field in fields if field.arb_id.upper().zfill(3) == right_id]
        for a in left:
            for b in right:
                per_recording = []
                for capture in held:
                    values, mask, _, step = align(capture, (a, b), step=.1, max_age=.3)
                    both = mask[:, 0] & mask[:, 1]
                    if both.sum() < 50 or min(np.std(values[both, 0]),
                                              np.std(values[both, 1])) == 0:
                        per_recording.append({"pairs": int(both.sum()), "correlation": None,
                                              "shift_control": None})
                        continue
                    x, y = values[both, 0], values[both, 1]
                    shift = max(1, len(values) // 3)
                    shifted_mask = mask[:, 0] & np.roll(mask[:, 1], shift)
                    shifted_x = values[shifted_mask, 0]
                    shifted_y = np.roll(values[:, 1], shift)[shifted_mask]
                    per_recording.append({"pairs": len(x), "step_seconds": step,
                                          "correlation": float(np.corrcoef(x, y)[0, 1]),
                                          "shift_control": (float(np.corrcoef(shifted_x, shifted_y)[0, 1])
                                                            if len(shifted_x) > 50 and
                                                            min(np.std(shifted_x), np.std(shifted_y)) > 0
                                                            else None)})
                checks.append({"left_id": left_id, "left_start": a.start,
                               "right_id": right_id, "right_start": b.start,
                               "held_out": per_recording})
    result["cross_signal_checks"] = checks
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vehicle", choices=ANCHORS)
    parser.add_argument("recordings", nargs=2, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.vehicle, args.recordings), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
