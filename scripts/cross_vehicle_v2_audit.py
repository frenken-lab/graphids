"""Audit the completed v2 pilot and emit a compact machine-readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_STEPS = {"silverado": 2592, "forester": 2736, "traverse": 2304}
EVALUATION_PARTITIONS = {
    "source_validation", "target_calibration", "target_normal_test",
    "target_known_attack", "target_unknown_attack",
}


def audit(root: Path, comparison: Path) -> dict:
    quality = json.loads(comparison.read_text())
    result = {"pilot": "cross_vehicle_v2", "quality": quality, "runs": {}}
    for fold, expected_steps in EXPECTED_STEPS.items():
        result["runs"][fold] = {}
        for variant in ("raw", "behavioral"):
            name = f"{fold}_{variant}_fit_v2"
            directory = root / name / name
            phases = json.loads((directory / ".graphids" / "phase_metrics.json").read_text())
            report = json.loads((directory / "cross_vehicle_report.json").read_text())
            train = [row for row in phases if row["phase"].startswith("train_epoch_")]
            validation = [row for row in phases if row["phase"].startswith("source_validation_epoch_")]
            complete = [row for row in phases if row["phase"] == "fit_complete"]
            evaluation = report["phase_metrics"]
            if len(train) != 2 or len(validation) != 2 or len(complete) != 1:
                raise ValueError(f"{name} lacks two training and validation epochs")
            if complete[0]["optimizer_steps"] != expected_steps:
                raise ValueError(f"{name} optimizer steps differ from complete lane plan")
            if {row["phase"] for row in evaluation} != EVALUATION_PARTITIONS:
                raise ValueError(f"{name} evaluation partition phases are incomplete")
            if train[0]["scored_events"] != train[1]["scored_events"]:
                raise ValueError(f"{name} training epochs cover different event counts")
            result["runs"][fold][variant] = {
                "optimizer_steps": expected_steps,
                "train_scored_events_per_epoch": train[0]["scored_events"],
                "train": train, "source_validation": validation,
                "evaluation": evaluation,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_root, args.comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
