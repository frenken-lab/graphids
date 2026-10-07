"""Compare three reference-conditioned CAN dynamics on frozen normal splits."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from can_discovery.preprocessing.capture import Capture, load_capture
from can_discovery.relationships.behavior.dynamics import (
    DynamicsConfig,
    fit_dynamics,
    score_events,
)

SETS = ("set_01", "set_02", "set_03", "set_04")
METHODS = ("timing", "phase", "change_coupling")


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_assigned(raw: Path, item: dict) -> Capture:
    path = raw / item["path"]
    capture = load_capture(path)
    if capture.sha256 != item["sha256"]:
        raise ValueError(f"frozen assignment hash changed: {path}")
    return capture


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _summary(capture: Capture, rows: list[dict], path: Path, profile) -> dict:
    hours = (capture.end - capture.start) / 3600
    alerts = {method: sum(row[method] > 0 for row in rows) for method in METHODS}
    return {"source": capture.source, "sha256": capture.sha256,
            "duration_seconds": capture.end - capture.start,
            "normal_exposure_hours": hours, "rows": capture.rows,
            "rules": {"timing": len(profile.timing), "phase": len(profile.phase),
                      "change_coupling": len(profile.change_coupling)},
            "alert_bins": alerts,
            "alert_bins_per_hour": {method: value / hours if hours else None
                                    for method, value in alerts.items()},
            "table": str(path)}


def _synthetic_deletion(capture: Capture, baseline: list[dict], profile, out: Path) -> dict:
    eligible = {rule.arb_id: rule for rule in profile.timing}
    counts = Counter(e.arb_id for e in capture.events if e.kind == "arrival" and e.arb_id in eligible)
    if not counts:
        return {"status": "no_eligible_timing_id", "synthetic": True}
    target = min(counts, key=lambda key: (-counts[key], key))
    occurrences = [e.timestamp for e in capture.events if e.kind == "arrival" and e.arb_id == target]
    onset = occurrences[len(occurrences) // 2]
    duration = min(10.0, max(profile.config["synthetic_min_seconds"],
                             10 * eligible[target].all_limit))
    stop = min(capture.end, onset + duration)
    removed = sum(e.kind == "arrival" and e.arb_id == target and onset <= e.timestamp < stop
                  for e in capture.events)
    # Reload retained raw frames so payload-change events after deletion are derived afresh.
    modified = load_capture(Path(capture.source), drop_interval=(target, onset, stop))
    rows = score_events(modified.events, capture.start, capture.end, profile)
    path = out / "synthetic_deletion.jsonl"
    _write_rows(path, rows)
    first = {}
    normal_alerts = {}
    synthetic_alerts = {}
    for method in METHODS:
        normal_alerts[method] = sum(row[method] > 0 and onset <= row["timestamp"] <= stop
                                    for row in baseline)
        synthetic_alerts[method] = sum(row[method] > 0 and onset <= row["timestamp"] <= stop
                                       for row in rows)
        incremental = next((row["timestamp"] - onset for before, row in zip(baseline, rows, strict=True)
                            if onset <= row["timestamp"] <= stop and row[method] > before[method]), None)
        first[method] = incremental
    return {"status": "scored", "synthetic": True, "target_id": target,
            "onset": onset, "end": stop, "removed_frames": removed,
            "baseline_alert_bins_in_interval": normal_alerts,
            "synthetic_alert_bins_in_interval": synthetic_alerts,
            "first_incremental_alert_delay_seconds": first,
            "table": str(path)}


def run_set(raw: Path, frozen: Path, out: Path, config: DynamicsConfig) -> dict:
    started = time.perf_counter()
    assignment = json.loads((frozen / "assignment.json").read_text())
    if {role: len(assignment[role]) for role in ("reference", "calibration", "normal_test")} != {
        "reference": 2, "calibration": 1, "normal_test": 1,
    }:
        raise ValueError(f"unexpected frozen assignment: {frozen}")
    for role in ("reference", "calibration", "normal_test"):
        for item in assignment[role]:
            if _hash(raw / item["path"]) != item["sha256"]:
                raise ValueError(f"frozen assignment hash changed: {raw / item['path']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "assignment.json").write_text(json.dumps(assignment, indent=2) + "\n")
    references = [_load_assigned(raw, item) for item in assignment["reference"]]
    fit_start = time.perf_counter()
    profile = fit_dynamics(references, config)
    fit_seconds = time.perf_counter() - fit_start
    (out / "profile.json").write_text(json.dumps(profile.to_dict(), indent=2) + "\n")
    del references
    results = {}
    for role in ("calibration", "normal_test"):
        capture = _load_assigned(raw, assignment[role][0])
        rows = score_events(capture.events, capture.start, capture.end, profile)
        path = out / f"{role}.jsonl"
        _write_rows(path, rows)
        results[role] = _summary(capture, rows, path, profile)
        if role == "normal_test":
            results["synthetic_deletion"] = _synthetic_deletion(capture, rows, profile, out)
    report = {"version": 1, "set": raw.name, "assignment": assignment,
              "settings": profile.config,
              "rule_counts": {"timing": len(profile.timing), "phase": len(profile.phase),
                              "change_coupling": len(profile.change_coupling)},
              "rejected_reasons": dict(Counter(item["reason"] for item in profile.rejected)),
              "results": results, "fit_seconds": fit_seconds,
              "runtime_seconds": time.perf_counter() - started,
              "notes": ["All definitions fitted on the two frozen normal references only.",
                        "Synthetic deletion is an explicitly labeled controlled perturbation.",
                        "Normal-file session independence is not proven by hashes."]}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
