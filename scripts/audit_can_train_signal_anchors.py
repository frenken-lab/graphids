"""Audit paper-documented message IDs without treating them as field truth."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

# Lampe and Meng, Computers & Security 140 (2024), section 4.3.3.
PAPER = "https://backend.orbit.dtu.dk/ws/files/354413940/1-s2.0-S0167404824000786-main.pdf"
ANCHORS = {
    "traverse": {"3E9": "displayed_speed", "0C9": "rpm", "1F5": "gear"},
    "impala": {"3E9": "displayed_speed", "1F5": "gear"},
    "silverado": {"135": "gear"},
    "forester": {"148": "gear"},
}


def audit(paths: list[Path], vehicle: str) -> dict:
    identifiers = ANCHORS[vehicle]
    recordings = []
    for path in paths:
        counts: Counter[str] = Counter()
        values: dict[str, set[str]] = defaultdict(set)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                identifier = row["arbitration_id"].upper().zfill(3)
                if identifier in identifiers and row.get("attack", "0") != "1":
                    counts[identifier] += 1
                    values[identifier].add(row["data_field"].upper())
        recordings.append({"path": str(path), "anchors": {
            identifier: {"role": role, "frames": counts[identifier],
                         "distinct_payloads": len(values[identifier])}
            for identifier, role in identifiers.items()}})
    return {"vehicle": vehicle, "source": PAPER,
            "evidence_level": "paper_message_identity_only", "recordings": recordings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vehicle", choices=ANCHORS)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.captures, args.vehicle), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
