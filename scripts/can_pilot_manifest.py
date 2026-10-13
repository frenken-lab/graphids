"""Freeze a small CTAT capture-level pilot after full upstream correspondence.

Requires the sequence audit and downloaded, commit-pinned upstream files.
Separate acquisition captures are sessions for this pilot; driving-trip or
day-level independence is not claimed. No raw cleaning or clock sorting occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ASSIGNMENTS = {
    "set_01/train_01_attack_free/accessory-1.csv": "train_reference",
    "set_01/train_01_attack_free/accessory-2.csv": "val_reference",
    "set_01/train_01_attack_free/attack-free-1.csv": "train_query",
    "set_01/train_01_attack_free/attack-free-2.csv": "val_query",
    "set_01/train_02_with_attacks/DoS-1.csv": "train_query",
    "set_01/train_02_with_attacks/DoS-2.csv": "val_query",
    "set_04/train_01_attack_free/attack-free-4.csv": "target_reference",
    "set_04/train_01_attack_free/accessory-4.csv": "target_reference",
    "set_04/train_01_attack_free/accessory-3.csv": "target_calibration",
    "set_04/train_01_attack_free/attack-free-3.csv": "target_normal_test",
    "set_02/test_04_unknown_vehicle_unknown_attack/DoS-3.csv": "target_attack_test",
    "set_02/test_04_unknown_vehicle_unknown_attack/DoS-4.csv": "target_attack_test",
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare_log(local, source):
    """Verify the complete timed-frame multiset under documented curation.

    Upstream label.py sorts clocks, including ties. Record original backsteps
    and ordering changes explicitly; never change local recording order. Times
    are compared at candump's six-decimal seconds precision.
    """
    with local.open(newline="") as raw, source.open() as original:
        first = original.readline().split()[0]
        origin = float(first.strip("()"))
        original.seek(0)
        reader = csv.DictReader(raw)
        first_local = next(reader)
        local_origin = float(first_local["timestamp"])
        raw.seek(0)
        local_frames = Counter(
            (
                round(float(row["timestamp"]), 6),
                row["arbitration_id"],
                row["data_field"],
            )
            for row in csv.DictReader(raw)
        )
        rows = sum(local_frames.values())
        offset = origin - local_origin
        end = last = origin
        backsteps = 0
        largest_backstep = 0.0
        original_count = 0
        for line in original:
            timestamp, _interface, frame = line.split()
            timestamp = float(timestamp.strip("()"))
            arb, payload = frame.split("#")
            key = (round(timestamp - offset, 6), arb, payload)
            if not local_frames[key]:
                raise ValueError(
                    f"timed frame differs at original row {original_count + 1}: {local}"
                )
            local_frames[key] -= 1
            if not local_frames[key]:
                del local_frames[key]
            backsteps += timestamp < last
            largest_backstep = max(largest_backstep, last - timestamp)
            last = timestamp
            end = max(end, timestamp)
            original_count += 1
        if local_frames or original_count != rows:
            raise ValueError(f"row count differs: {local}")
    return {
        "rows": rows,
        "acquisition_start": origin,
        "acquisition_end": end,
        "original_decreasing_clock_rows": backsteps,
        "original_largest_backstep_seconds": largest_backstep,
        "correspondence": "complete timed-frame multiset at 1 microsecond precision",
        "ordering_provenance": "curator label.py sorts timestamps; local order preserved",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.review_root
    review = json.loads((root / "recording-review.json").read_text())
    upstream = json.loads((root / "upstream/pilot-full-sources.json").read_text())
    by_hash = {r["sha256"]: r for r in review["recordings"]}
    inventory = {}
    canonical = {}
    for group in review["aliases"]:
        canonical_record = next(
            r for r in review["recordings"] if r["path"] == group[0]
        )
        for path in group:
            inventory[path] = {**by_hash[canonical_record["sha256"]], "path": path}
            canonical[path] = group[0]
    selected = {canonical[path] for path in ASSIGNMENTS}
    if any(set(pair["paths"]) <= selected for pair in review["candidate_reuse"]):
        raise ValueError(
            "selected captures have candidate sequence reuse; review first"
        )
    records, evidence = [], []
    for path, role in ASSIGNMENTS.items():
        rec = inventory[path]
        if rec["bad_clock"] or rec["bad_payload"] or rec["bad_labels"]:
            raise ValueError(f"invalid raw capture: {path}")
        local = args.raw_root / path
        if sha256(local) != rec["sha256"]:
            raise ValueError(f"local content changed: {path}")
        sources = [s for s in upstream if s["local"] == path]
        if any("error" in s for s in sources) or not sources:
            raise ValueError(f"upstream evidence unavailable: {path}")
        log = next(s for s in sources if s["source"].endswith(".log"))
        correspondence = compare_log(local, Path(log["download"]))
        normal = "/train_01_attack_free/" in path
        if normal and rec["positive_rows"]:
            raise ValueError(f"positive labels in declared normal acquisition: {path}")
        if not normal:
            labeled = next(s for s in sources if s["source"].endswith(".csv"))
            if sha256(labeled["download"]) != rec["sha256"]:
                raise ValueError(f"curator label file differs: {path}")
        dataset = path.split("/")[0]
        vehicle = log["source"].split("/")[0]
        key = f"{vehicle}/{local.stem}"
        records.append(
            {
                "key": key,
                "dataset": dataset,
                "vehicle": vehicle,
                "session": f"acquisition:{vehicle}:{correspondence['acquisition_start']:.6f}",
                "path": str(local.resolve()),
                "sha256": rec["sha256"],
                "role": role,
                "verified_normal": normal,
                "timestamp_unit": "seconds",
                "payload_length_verified": True,
                "labels_verified": True,
                "attack_intervals": [],
            }
        )
        evidence.append(
            {
                "key": key,
                "role": role,
                "vehicle": vehicle,
                **correspondence,
                "span_seconds": rec["span_seconds"],
                "upstream": [{**s, "sha256": sha256(s["download"])} for s in sources],
            }
        )
    for i, left in enumerate(evidence):
        for right in evidence[i + 1 :]:
            if left["vehicle"] == right["vehicle"] and max(
                left["acquisition_start"], right["acquisition_start"]
            ) <= min(left["acquisition_end"], right["acquisition_end"]):
                raise ValueError("original acquisition intervals overlap")
    (root / "pilot-provenance.json").write_text(
        json.dumps(
            {
                "reviewed_scope": "CTAT Impala-to-Forester normal/DoS capture pilot only",
                "session_definition": "distinct original acquisition capture, not driving trip/day",
                "limitations": [
                    "one underlying dataset family and one target vehicle",
                    "no independently trusted attack intervals; event/bin metrics only",
                    "sequence fingerprint sampling does not prove absence of all partial reuse",
                    "target calibration is accessory mode, with fewer observed IDs",
                    "payload byte counts preserve candump serialization, not independent DLC",
                ],
                "recordings": evidence,
            },
            indent=2,
        )
    )
    manifest = {"version": 1, "provenance_reviewed": True, "records": records}
    (root / "can-recording-manifest-v1.json").write_text(json.dumps(manifest, indent=2))
    print(f"reviewed {len(records)} disjoint acquisition captures")


if __name__ == "__main__":
    main()
