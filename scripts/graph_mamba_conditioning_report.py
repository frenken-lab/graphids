"""Render a compact human-readable report from completed frozen-weight diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def assemble(root):
    report = json.loads((root / "diagnostics-v2.json").read_text())
    for name, trial in report["trials"].items():
        base = (
            Path(report.get("reused_baseline_root", root))
            if name == "full-reference"
            else root
        )
        artifact_root = base / name / "artifacts"
        trial["artifact_root"] = str(artifact_root)
        trial["coverage"] = json.loads((artifact_root / "coverage.json").read_text())
        trial["target_bundle"] = json.loads(
            (artifact_root / "target-calibration-v2.json").read_text()
        )
        coverage = trial["coverage"]
        assert (
            coverage["complete"]
            and coverage["scored_bins"] == coverage["expected_bins"]
        )
        assert coverage["missing_bins"] == coverage["duplicate_bins"] == 0
        assert set(trial["bundle"]) == {"zero-shot", "conditioned"}
        for field in (
            "center",
            "scale",
            "available",
            "checkpoint_hash",
            "source_manifest",
            "reference_policy",
        ):
            assert (
                trial["target_bundle"][field] == trial["bundle"]["conditioned"][field]
            ), field
        for track in ("source", "strict", "target_calibrated"):
            for group in trial[track]["groups"].values():
                for field in ("bin_alerts", "normal_bin_alerts"):
                    assert (
                        sum(
                            group[head][field]
                            for head in (
                                "classification_only",
                                "anomaly_only",
                                "overlap",
                            )
                        )
                        == group["combined"][field]
                    )
        for key, group in trial["target_calibrated"]["groups"].items():
            assert group["heads"] == trial["strict"]["groups"][key]["heads"]
    curve_path = root.parent / "corrected-curve/corrected-learning-curve-v2.json"
    if curve_path.exists():
        curve = json.loads(curve_path.read_text())
        if "actual_final_target" not in curve:
            raise ValueError("retained-checkpoint replay is still incomplete")
        report["last_checkpoint_file_replay"] = report.get("last_checkpoint")
        report["last_checkpoint"] = curve["actual_final_target"]
        report["last_checkpoint_source"] = curve["epochs"][-1]["source"]
        report["last_checkpoint_target_calibrated"] = curve[
            "actual_final_target_calibrated"
        ]
        report["corrected_source_curve"] = curve
        final = curve["epochs"][-1]
        nominal_last = Path(final["path"]).parent.parent / "checkpoints/last.ckpt"
        last_hash = hashlib.sha256(nominal_last.read_bytes()).hexdigest()
        report["checkpoint_label_audit"] = dict(
            nominal_last_path=str(nominal_last),
            nominal_last_sha256=last_hash,
            nominal_last_is_selected_copy=last_hash == report["checkpoint_sha256"],
            actual_final_path=final["path"],
            actual_final_sha256=final["sha256"],
            actual_final_epoch=final["displayed_epoch"],
        )
    for name, filename in (
        ("verification", "verification-v2.json"),
        ("replay_invariants", "replay-invariants-v2.json"),
    ):
        if (root / filename).exists():
            report[name] = json.loads((root / filename).read_text())
    report["localization"] = (
        "At selected epoch 4, persistent reference anchors, cross-ID graph messages and classifier pooling have interacting effects. Expiry restores target classification ranking but worsens source anomaly AP and target false-bin alerts. This does not identify a unique cause or select a target winner."
    )
    report["source_recommendations"] = [
        "Prioritize profile and graph-message ablations for broader source validation: both improve the source minimum-of-four AP criterion.",
        "Do not adopt expiry from target classification AP: source anomaly AP declines.",
        "Keep the original epoch-4 selection. Use the corrected source-only checkpoint criterion and independent final-weight saving in the next run.",
        "Review additional source vehicles and create a new independent holdout before generalization claims.",
    ]
    (root / "final-report-v2.json").write_text(json.dumps(report, indent=2))
    return report


def render(root):
    report_path = root / "final-report-v2.json"
    if not report_path.exists():
        report_path = root / "diagnostics-v2.json"
    report = json.loads(report_path.read_text())
    saved = json.loads((root / "saved-scores-corrected-v2.json").read_text())
    lines = [
        "Graph-Mamba corrected evaluation and reference diagnostics, version 2",
        "",
        f"Checkpoint file SHA256: {report['checkpoint_sha256']}",
        "Frozen epoch-4 weights; original source zero-shot classification selection is unchanged.",
        "Forester is diagnostic evidence. These results do not constitute a new independent holdout.",
        "",
        "CORRECTED RETAINED SCORES (historical pooled normalization)",
    ]

    def ap(group, head):
        v = group["heads"][head]["ap"]
        return f"{v:.8f}" if v is not None else "unavailable (missing both classes)"

    for key, group in saved["groups"].items():
        if "/vehicle/" in key:
            lines.append(
                f"{key}: classification AP={ap(group, 'classification')}; anomaly AP={ap(group, 'anomaly')}"
            )
    lines += [
        "Historical anomaly AP was affected by automatic sigmoid. Registry-specific vehicle AP also omitted normal recordings from other registry names.",
        "Retained snapshots are replayed separately; original curve values remain historical. Corrected source curves are provided only for weights actually retained.",
        "",
        "SOURCE-REPLAYED BASELINES (separate calibration for each mode)",
    ]
    full = report["trials"]["full-reference"]
    for phase in ("source", "strict", "target_calibrated"):
        for key, group in full[phase]["groups"].items():
            if phase == "target_calibrated" and key.startswith("zero-shot/"):
                continue
            if "/vehicle/" in key:
                lines.append(
                    f"{phase} {key}: classification AP={ap(group, 'classification')}; anomaly AP={ap(group, 'anomaly')}; scored bins={group['scored_bins']}; normal hours={group['normal_hours']:.8f}; eligible cls/anomaly={group['heads']['classification']['eligible_bins']}/{group['heads']['anomaly']['eligible_bins']}; positive prevalence={group['heads']['classification']['positive_prevalence']:.6f}"
                )
    lines += [
        "",
        "PREDECLARED CONDITIONING EFFECTS (variant minus full reference on identical eligible bins)",
        "Zero-shot is the reference-free baseline in the full-reference trial; the other seven trials use the same frozen weights.",
        "variant | source classification delta | source anomaly delta | target classification delta | target anomaly delta | anomaly eligibility +/-",
    ]
    for name, effect in report["effects"].items():
        source = report["source_effects"][name]
        lines.append(
            f"{name} | {source['classification']['delta_ap']:+.8f} | {source['anomaly']['delta_ap']:+.8f} | {effect['classification']['delta_ap']:+.8f} | {effect['anomaly']['delta_ap']:+.8f} | {effect['anomaly']['variant_only_eligible']}/{effect['anomaly']['baseline_only_eligible']}"
        )
    lines += [
        "",
        "LIFECYCLE COMPONENT AVAILABILITY (all test bins, including unknown labels)",
    ]
    lifecycle = (
        report["effects"]
        .get("no-persistent-anchors", {})
        .get("component_eligibility", {})
    )
    for component, counts in lifecycle.items():
        lines.append(
            f"component {component}: baseline available={counts['baseline_available_bins']}, expiry available={counts['variant_available_bins']}, newly available={counts['variant_only_available_bins']}, no longer available={counts['baseline_only_available_bins']}"
        )
    lines += [
        "",
        "TARGET BIN ALERTS (counts and verified-normal exposure, not incidents)",
        "variant / track / mode | class-only | anomaly-only | overlap | combined | normal class/h | normal anomaly/h | normal combined/h",
    ]
    for name, trial in report["trials"].items():
        for track in ("strict", "target_calibrated"):
            for key, group in trial[track]["groups"].items():
                if "/vehicle/" not in key:
                    continue
                if track == "target_calibrated" and key.startswith("zero-shot/"):
                    continue
                parts = [
                    str(group[h]["bin_alerts"])
                    for h in (
                        "classification_only",
                        "anomaly_only",
                        "overlap",
                        "combined",
                    )
                ]
                parts += [
                    f"{group[h]['false_bin_alerts_per_hour']:.3f}"
                    if group[h]["false_bin_alerts_per_hour"] is not None
                    else "unavailable"
                    for h in ("classification", "anomaly", "combined")
                ]
                lines.append(
                    f"{name} / {track} / {key.split('/')[0]} | " + " | ".join(parts)
                )
                if track == "target_calibrated":
                    assert group["heads"] == trial["strict"]["groups"][key]["heads"], (
                        "threshold-only target calibration changed AP"
                    )
    lines += [
        "",
        "SOURCE-ONLY BALANCED DIAGNOSTIC CRITERION (minimum of four vehicle-macro APs)",
    ]
    import statistics

    balanced = {}
    for name, trial in report["trials"].items():
        means = []
        for mode in ("zero-shot", "conditioned"):
            for head in ("classification", "anomaly"):
                values = [
                    group["heads"][head]["ap"]
                    for key, group in trial["source"]["groups"].items()
                    if key.startswith(mode + "/vehicle/")
                    and group["heads"][head]["ap"] is not None
                ]
                if values:
                    means.append(statistics.mean(values))
        if len(means) == 4:
            balanced[name] = min(means)
            lines.append(f"{name}: {balanced[name]:.8f}")
    candidates = [
        name
        for name, value in balanced.items()
        if value > balanced["full-reference"] and name != "full-reference"
    ]
    lines.append(
        "Candidates for broader SOURCE validation under this criterion: "
        + (", ".join(candidates) if candidates else "none")
    )
    lines.append(
        "This prioritization uses source validation only. Do not choose a production intervention from Forester results."
    )
    lines += ["", "CALIBRATION THRESHOLDS (classification / anomaly)"]
    for name, trial in report["trials"].items():
        for mode, bundle in trial["bundle"].items():
            lines.append(f"{name} source-normal {mode}: {bundle['thresholds']}")
        if "target_bundle" in trial:
            lines.append(
                f"{name} target-normal conditioned: {trial['target_bundle']['thresholds']}"
            )
    if "corrected_source_curve" in report:
        lines += [
            "",
            "CORRECTED SOURCE CURVE FROM RETAINED WEIGHTS",
            "epoch | zero-shot classification | zero-shot anomaly | conditioned classification | conditioned anomaly",
        ]
        for epoch in report["corrected_source_curve"]["epochs"]:
            values = []
            for mode in ("zero-shot", "conditioned"):
                for head in ("classification", "anomaly"):
                    scores = [
                        group["heads"][head]["ap"]
                        for key, group in epoch["source"]["groups"].items()
                        if key.startswith(mode + "/vehicle/")
                        and group["heads"][head]["ap"] is not None
                    ]
                    values.append(
                        f"{statistics.mean(scores):.8f}" if scores else "unavailable"
                    )
            lines.append(f"{epoch['displayed_epoch']} | " + " | ".join(values))
        lines.append(
            "Epoch 4 remains the original selected checkpoint; this corrected curve does not retrospectively select a new checkpoint."
        )
    if "last_checkpoint_source" in report:
        last_source = report["last_checkpoint_source"]
        lines += ["", "CORRECTED RETAINED LAST CHECKPOINT"]
        for phase, metrics in (
            ("source", last_source),
            ("target", report.get("last_checkpoint", {})),
        ):
            for key, group in metrics.get("groups", {}).items():
                if "/vehicle/" in key:
                    lines.append(
                        f"last {phase} {key}: classification AP={ap(group, 'classification')}; anomaly AP={ap(group, 'anomaly')}"
                    )
    lines += [
        "",
        "CALIBRATION AND REPRODUCIBILITY",
        "Each policy replays source validation, then freezes per-mode component centers/scales and head thresholds before target replay.",
        "Quantile=0.999; scale floor=1e-4. Classification includes first and terminal scored normal bins. Forecast components require forecast eligibility and availability masks.",
        "Target calibration fits only two conditioned thresholds from target_calibration. It retains source normalization; scores and AP are unchanged. Zero-shot uses source calibration only.",
        "AP excludes target_calibration and unknown labels (-1). Candidate bin alerts include all scored test bins; normal exposure uses verified-normal recordings.",
        "Bundle checkpoint_hash is SHA256 of learned parameter names and values, excluding calibration buffers. The checkpoint file hash above identifies the original file.",
        "Per-recording/label distributions, component masks, coverage, live/silent nodes, active edges and raw scores are retained in each trial directory.",
        "Policy changes reset model state; lifecycle changes rebuild snapshots. Comparison asserts unchanged recording/bin coverage, timestamps, exposure, labels, normality and event counts.",
        "",
        "INTERPRETATION AND NEXT ACTIONS",
        "Source validation, rather than Forester performance, governs any next-run recommendation. Review the source deltas above before changing conditioning.",
        "These one-at-a-time interventions can include interactions and do not prove a unique cause. Expiry changes neighborhoods, lifecycle and pooling together, leaving their interactions unresolved.",
        "Keep the current weights and epoch-4 selection unchanged. The separate corrected next-run preset maximizes the minimum of source vehicle-macro AP over both heads and modes; ties retain earlier checkpoints.",
        "Additional reviewed vehicles and a new independent holdout are required before broader generalization claims. Architecture redesign and retraining remain subsequent decisions.",
        "",
        f"Machine-readable evidence: {report_path}",
    ]
    lines += [
        "",
        report.get("localization", ""),
        *report.get("source_recommendations", []),
    ]
    if "checkpoint_label_audit" in report:
        audit = report["checkpoint_label_audit"]
        lines += [
            "",
            "CHECKPOINT LABEL AUDIT",
            "Original last.ckpt is byte-identical to epoch-4 best.ckpt; it is not the final epoch.",
            f"Actual final weights recovered from {audit['actual_final_path']}",
            "All ten epoch snapshots were retained by Ray. Their corrected source curves are linked in the JSON report; no intermediate values were invented.",
            "The future preset uses a separate unmonitored epoch checkpoint so last weights are retained even when monitored AP declines.",
            "The recovered epoch-10 conditioned classifier AP is high without an intervention. The severe ranking failure is specific to the selected epoch-4 weights; target results do not choose a replacement checkpoint.",
        ]
    if "verification" in report:
        lines += ["", "VERIFICATION", json.dumps(report["verification"], indent=2)]
    lines += [
        "",
        "INTERVENTION DEFINITIONS",
        "Recent classifier pooling means reference IDs observed in the current bin (count > 0); all live transient nodes remain eligible. Forecast pooling is unchanged.",
    ]
    (root / "report-v2.txt").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    assemble(args.root.resolve())
    render(args.root.resolve())


if __name__ == "__main__":
    main()
