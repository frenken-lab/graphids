"""Combine the six leave-one-vehicle-out pilot reports without retuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FOLDS = ("silverado", "forester", "traverse")
VARIANTS = ("raw", "behavioral")
ATTACK_SETS = ("target_known_attack", "target_unknown_attack")


def load_reports(directory: Path) -> dict[str, dict[str, dict]]:
    reports = {}
    for fold in FOLDS:
        reports[fold] = {}
        for variant in VARIANTS:
            name = f"{fold}_{variant}_fit"
            path = directory / name / name / "cross_vehicle_report.json"
            report = json.loads(path.read_text())
            if Path(report["config"]).stem != name:
                raise ValueError(f"report/config mismatch: {path}")
            reports[fold][variant] = report
    return reports


def summarize(reports: dict[str, dict[str, dict]]) -> dict:
    summary: dict = {"folds": {}, "pilot_rule": {
        "auprc_min_absolute_gain": 0.02,
        "required_gain_folds": 2,
        "normal_false_alerts_per_hour_limit": "max(raw * 1.25, raw + 1)",
        "required_recall_gain_folds": 2,
    }}
    for regime in ("zero_shot", "normal_target_adaptation"):
        ap_gains = recall_gains = 0
        alarm_ok = True
        summary["folds"][regime] = {}
        for fold in FOLDS:
            raw = reports[fold]["raw"][regime]
            behavioral = reports[fold]["behavioral"][regime]
            raw_ap = sum(raw[name]["auprc"] for name in ATTACK_SETS) / len(ATTACK_SETS)
            beh_ap = sum(behavioral[name]["auprc"] for name in ATTACK_SETS) / len(ATTACK_SETS)
            raw_rec = sum(raw[name]["recall"] for name in ATTACK_SETS) / len(ATTACK_SETS)
            beh_rec = sum(behavioral[name]["recall"] for name in ATTACK_SETS) / len(ATTACK_SETS)
            raw_alarm = raw["target_normal_test"]["false_alerts_per_hour"]
            beh_alarm = behavioral["target_normal_test"]["false_alerts_per_hour"]
            if any(value is None for value in (raw_ap, beh_ap, raw_rec, beh_rec, raw_alarm, beh_alarm)):
                raise ValueError(f"missing primary metric for {fold} {regime}")
            ap_gain = beh_ap - raw_ap
            recall_gain = beh_rec - raw_rec
            alarm_limit = max(raw_alarm * 1.25, raw_alarm + 1)
            ap_gains += ap_gain >= 0.02
            recall_gains += recall_gain >= 0
            alarm_ok &= beh_alarm <= alarm_limit
            summary["folds"][regime][fold] = {
                "raw_macro_auprc": raw_ap, "behavioral_macro_auprc": beh_ap,
                "auprc_gain": ap_gain, "raw_macro_recall": raw_rec,
                "behavioral_macro_recall": beh_rec, "recall_gain": recall_gain,
                "raw_normal_false_alerts_per_hour": raw_alarm,
                "behavioral_normal_false_alerts_per_hour": beh_alarm,
                "alarm_limit": alarm_limit,
            }
        summary[regime] = {
            "auprc_gain_folds": ap_gains,
            "recall_gain_folds": recall_gains,
            "normal_alarm_rule_passed": alarm_ok,
            "pilot_rule_passed": ap_gains >= 2 and recall_gains >= 2 and alarm_ok,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = summarize(load_reports(args.run_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
