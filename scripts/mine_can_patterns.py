"""Mine raw CAN relationships and export standalone tracking diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from graphids.can_patterns import (
    MinerConfig,
    PatternMiner,
    PatternTracker,
    TrackerState,
    load_capture,
)
from graphids.core.data.datasets.can_bus import declared_vehicle_id
from graphids.paths import lake_root

SETS = ("set_01", "set_02", "set_03", "set_04")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def assign(raw: Path) -> dict:
    normal = sorted((raw / "train_01_attack_free").glob("*.csv"))
    if len(normal) != 4:
        raise ValueError(f"expected four normal files under {raw}")
    accessory = [p for p in normal if "accessory" in p.stem]
    other = [p for p in normal if p not in accessory]
    if len(accessory) == 2 and len(other) == 2:
        roles = {"reference": [accessory[0], other[0]], "calibration": [accessory[1]],
                 "normal_test": [other[1]]}
    else:
        roles = {"reference": normal[:2], "calibration": normal[2:3], "normal_test": normal[3:]}
    assignment = {role: [{"path": str(p.relative_to(raw)), "sha256": file_hash(p)} for p in paths]
                  for role, paths in roles.items()}
    hashes = [item["sha256"] for files in assignment.values() for item in files]
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"normal roles contain duplicate files: {raw}")
    return assignment


def id_timeout_limits(captures) -> dict[str, float]:
    gaps = defaultdict(list)
    for capture in captures:
        previous = {}
        for event in capture.events:
            if event.kind != "arrival":
                continue
            if event.arb_id in previous and event.timestamp > previous[event.arb_id]:
                gaps[event.arb_id].append(event.timestamp - previous[event.arb_id])
            previous[event.arb_id] = event.timestamp
    return {key: 3 * statistics.median(values) for key, values in gaps.items() if values}


def score_capture(path: Path, catalog, output: Path, timeout_limits: dict[str, float]) -> dict:
    start_clock = time.perf_counter()
    capture = load_capture(path)
    width = catalog.settings["bin_width"]
    state = TrackerState()
    last_id = {}
    timeout_active = set()
    pattern_alerts = timeout_alerts = completed = overdue = 0
    covered_ids = {step.split("|")[0] for pattern in catalog.patterns for step in pattern.steps}
    covered_rows = sum(e.kind == "arrival" and e.arb_id in covered_ids for e in capture.events)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        if capture.events:
            origin = capture.start
            bin_number = 0
            batch = []
            def emit(close, events):
                nonlocal state, pattern_alerts, timeout_alerts, completed, overdue
                features, scores, state = PatternTracker.step(events, catalog, state, close)
                for event in events:
                    if event.kind == "arrival":
                        last_id[event.arb_id] = event.timestamp
                        timeout_active.discard(event.arb_id)
                timed_out = []
                for key, stamp in last_id.items():
                    if key in timeout_limits and key not in timeout_active and close > stamp + timeout_limits[key]:
                        timeout_active.add(key)
                        timed_out.append(key)
                features["id_timeout"] = [int(key in timed_out) for key in catalog.id_keys] + [0] * (catalog.settings["max_streams"] - len(catalog.id_keys))
                scores["id_timeout_alert"] = int(bool(timed_out))
                pattern_alerts += scores["pattern_alert"]
                timeout_alerts += scores["id_timeout_alert"]
                completed += sum(features["relationship_completed"])
                overdue += sum(features["relationship_overdue"])
                handle.write(json.dumps({"features": features, "scores": scores}, separators=(",", ":")) + "\n")
            for event in capture.events:
                while event.timestamp > origin + (bin_number + 1) * width:
                    emit(origin + (bin_number + 1) * width, batch)
                    batch = []
                    bin_number += 1
                batch.append(event)
            emit(capture.end, batch)
    hours = (capture.end - capture.start) / 3600
    return {"source": str(path), "sha256": capture.sha256, "rows": capture.rows,
            "duration_seconds": capture.end - capture.start, "normal_exposure_hours": hours,
            "payload_lengths": capture.payload_lengths,
            "decreasing_timestamps": capture.decreasing_timestamps,
            "invalid_payloads": capture.invalid_payloads,
            "raw_payload_chars": capture.raw_payload_chars, "covered_rows": covered_rows,
            "coverage_rate": covered_rows / capture.rows if capture.rows else 0,
            "pattern_completions": completed, "pattern_overdue": overdue,
            "pattern_alert_bins": pattern_alerts, "id_timeout_alert_bins": timeout_alerts,
            "pattern_alert_bins_per_hour": pattern_alerts / hours if hours else None,
            "id_timeout_alert_bins_per_hour": timeout_alerts / hours if hours else None,
            "runtime_seconds": time.perf_counter() - start_clock,
            "table": str(output)}


def run_set(raw: Path, out: Path, config: MinerConfig, include_attacks: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    assignment = assign(raw)
    (out / "assignment.json").write_text(json.dumps(assignment, indent=2) + "\n")
    started = time.perf_counter()
    references = [load_capture(raw / item["path"]) for item in assignment["reference"]]
    catalog = PatternMiner.fit(references, config)
    (out / "catalog.json").write_text(json.dumps(catalog.to_dict(), indent=2) + "\n")
    limits = id_timeout_limits(references)
    results = {}
    for role in ("calibration", "normal_test"):
        results[role] = [score_capture(raw / item["path"], catalog,
                         out / "tables" / (role + "_" + Path(item["path"]).stem + ".jsonl"), limits)
                         for item in assignment[role]]
    results["attacks"] = []
    if include_attacks:
        for folder in sorted(raw.glob("test_*")):
            if not folder.is_dir():
                continue
            # The first four test folders have explicit vehicle identities. The
            # mixed suppression/masquerade folders remain diagnostic captures.
            if folder.name.startswith(("test_01_", "test_02_", "test_03_", "test_04_")) and declared_vehicle_id(raw.name, folder.name) != declared_vehicle_id(raw.name, "train_01_attack_free"):
                continue
            for path in sorted(folder.rglob("*.csv")):
                label = str(path.relative_to(raw)).replace("/", "__")
                result = score_capture(path, catalog, out / "tables" / (label + ".jsonl"), limits)
                result["interpretation"] = "diagnostic_unlabeled" if "suppress" in folder.name else "diagnostic"
                results["attacks"].append(result)
    report = {"version": 1, "set": raw.name, "assignment": assignment,
              "settings": asdict(config), "patterns": len(catalog.patterns),
              "pattern_stability": [{"steps": p.steps, "reference_support": p.reference_support,
                                     "completion_rate": p.completion_rate, "uncertainty": p.uncertainty}
                                    for p in catalog.patterns],
              "candidate_reasons": dict(Counter(c["reason"] for c in catalog.candidates)),
              "candidates": catalog.candidates,
              "results": results, "runtime_seconds": time.perf_counter() - started,
              "notes": ["Reference-conditioned pilot; whole IDs only.",
                        "Suppression traces are diagnostic: absent-frame event recall is undefined without intervals.",
                        "Normal file independence is not established by filename or exact hash."]}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normal-only", action="store_true", help="skip attack diagnostics")
    parser.add_argument("--sets", nargs="+", choices=SETS, default=SETS)
    args = parser.parse_args()
    root = args.raw_root or Path(lake_root()) / "raw"
    for name in args.sets:
        report = run_set(root / name, args.output / name,
                         MinerConfig(), not args.normal_only)
        print(f"{name}: {report['patterns']} patterns; {report['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
