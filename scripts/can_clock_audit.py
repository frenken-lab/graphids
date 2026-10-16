"""Measure raw prefix-clock corrections and replay per-interface clock integrity."""

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.review_root
    plan = json.loads((root / "plan.json").read_text())
    sources = json.loads((root / "upstream/sources.json").read_text())
    records = []
    for path, item in plan["assignments"].items():
        if item.get("source_log_repo") != "can-ml":
            continue
        maximum = Decimal("-Infinity")
        changed, largest, examples = 0, Decimal(0), []
        with (args.raw_root / path).open() as handle:
            for line, row in enumerate(csv.DictReader(handle), 2):
                timestamp = Decimal(row["timestamp"])
                delta = max(maximum - timestamp, Decimal(0))
                if delta:
                    changed += 1
                    if len(examples) < 3 and delta > Decimal(".001"):
                        examples.append(
                            dict(
                                line=line,
                                timestamp=str(timestamp),
                                correction=str(delta),
                            )
                        )
                largest = max(largest, delta)
                maximum = max(maximum, timestamp)
        last, backsteps, corrections = {}, {}, {}
        log = next(
            s for s in sources if s["local"] == path and s["source"].endswith(".log")
        )
        with Path(log["download"]).open() as handle:
            for row in handle:
                timestamp, interface, _ = row.split()
                value = Decimal(timestamp.strip("()"))
                delta = max(last.get(interface, value) - value, Decimal(0))
                if delta:
                    backsteps[interface] = backsteps.get(interface, 0) + 1
                corrections[interface] = max(
                    corrections.get(interface, Decimal(0)), delta
                )
                last[interface] = max(last.get(interface, value), value)
        record = dict(
            path=path,
            changed_rows=changed,
            maximum_correction_seconds=str(largest),
            examples_over_1ms=examples,
            interface_backsteps=backsteps,
            interface_maximum_corrections={k: str(v) for k, v in corrections.items()},
        )
        records.append(record)
        (root / "clock-audit.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
