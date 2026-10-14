"""Frozen checkpoint diagnostics. Run in Slurm; never selects an intervention on target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import lightning.pytorch as pl
import torch

from graphids.core.data.snapshots.datamodule import CANSnapshotDataModule
from graphids.core.data.snapshots.policy import INTERVENTIONS
from graphids.core.models.calibration import CalibrationBundle, weights_digest
from graphids.core.models.graph_mamba import VehicleGraphMambaJoint
from graphids.core.models.score_columns import ScoreTable


def load_scores(path):
    with path.open() as handle:
        return ScoreTable.from_rows(json.loads(line) for line in handle)


def saved_report(path, output):
    table = load_scores(path)
    table.check_unique("saved")
    report = table.report()
    report.update(
        source_scores=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        calibration="historical pooled normalization; AP corrected from retained scores; thresholds require source replay",
        diagnostic_columns_unavailable=["reference_coverage", "active_edges"],
        historical_anomaly_ap="affected by automatic sigmoid; intermediate epochs cannot be reconstructed without retained weights/scores",
    )
    (output / "saved-scores-corrected-v2.json").write_text(json.dumps(report, indent=2))


def compare(table, baseline, mode="conditioned"):
    # Context digests differ across policies; match recording + bin and mode.
    def index(t, selected_mode):
        return {
            (r["recording"], r["bin"]): r
            for r in t
            if (r["context"] == "zero-shot") == (selected_mode == "zero-shot")
            and r["role"] in ("target_normal_test", "target_attack_test", "val_query")
        }

    a, b = index(table, mode), index(baseline, "conditioned")
    if set(a) != set(b):
        raise AssertionError("intervention changed scored bin coverage")
    result = {}
    from graphids.core.models.score_columns import raw_average_precision

    for head in ("classification", "anomaly"):
        common = [
            key
            for key in a
            if a[key]["label"] >= 0
            and (
                head != "anomaly"
                or (a[key]["anomaly_eligible"] and b[key]["anomaly_eligible"])
            )
        ]
        labels = [a[k]["label"] for k in common]
        av = raw_average_precision([a[k][head] for k in common], labels)
        bv = raw_average_precision([b[k][head] for k in common], labels)
        result[head] = dict(
            common_eligible_bins=len(common),
            variant_ap=av,
            baseline_ap=bv,
            delta_ap=av - bv if av is not None and bv is not None else None,
            variant_only_eligible=sum(
                a[k]["anomaly_eligible"] and not b[k]["anomaly_eligible"] for k in a
            )
            if head == "anomaly"
            else 0,
            baseline_only_eligible=sum(
                b[k]["anomaly_eligible"] and not a[k]["anomaly_eligible"] for k in a
            )
            if head == "anomaly"
            else 0,
        )
    result["component_eligibility"] = {
        str(i): {
            "variant_available_bins": sum(
                bool(a[k]["anomaly_component_mask"][i]) for k in a
            ),
            "baseline_available_bins": sum(
                bool(b[k]["anomaly_component_mask"][i]) for k in a
            ),
            "variant_only_available_bins": sum(
                bool(a[k]["anomaly_component_mask"][i])
                and not b[k]["anomaly_component_mask"][i]
                for k in a
            ),
            "baseline_only_available_bins": sum(
                bool(b[k]["anomaly_component_mask"][i])
                and not a[k]["anomaly_component_mask"][i]
                for k in a
            ),
        }
        for i in range(3)
    }
    for field in ("close", "exposure", "label", "normal", "events"):
        if any(a[k][field] != b[k][field] for k in a):
            raise AssertionError(f"intervention changed {field}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-scores", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--last-checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saved-only", action="store_true")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Reuse a completed full-reference trial; raw artifacts stay immutable",
    )
    parser.add_argument("--variants", nargs="+", choices=list(INTERVENTIONS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved_report(args.saved_scores, args.output)  # always precedes new inference
    if args.saved_only:
        return
    if not all((args.checkpoint, args.manifest, args.cache_root)):
        parser.error("checkpoint, manifest and cache-root are required for inference")
    curve_path = args.checkpoint.parent.parent / "artifacts/learning-curve.json"
    if curve_path.exists():
        curve = json.loads(curve_path.read_text())
        curve["metric_audit_v2"] = {
            "historical_anomaly_ap": "affected by automatic sigmoid; values below are retained historical values, not corrected intermediate epochs",
            "retained_weights_recomputed_separately": ["best", "last"],
        }
        (args.output / "historical-learning-curve-annotated-v2.json").write_text(
            json.dumps(curve, indent=2)
        )
    torch.set_num_threads(2)
    report = dict(
        version=2,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        implementation_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        worktree_status=subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for base in ("graphids", "scripts")
            for path in sorted(Path(base).rglob("*.py"))
        },
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        original_selection="epoch 4; source zero-shot classification AP; unchanged",
        target_status="diagnostic evidence; not independent holdout",
        trials={},
        effects={},
        source_effects={},
        interpretation="Intervention effects can include interactions; target does not select a winner.",
    )
    baseline = source_baseline = None
    dm = None
    if args.baseline_root:
        baseline = load_scores(
            args.baseline_root / "full-reference/artifacts/bin-scores.jsonl"
        )
        source_baseline = load_scores(
            args.baseline_root / "full-reference/source-scores-v2.jsonl"
        )
        prior = json.loads((args.baseline_root / "diagnostics-v2.json").read_text())
        report["trials"]["full-reference"] = prior["trials"]["full-reference"]
        report["trials"]["full-reference"]["target_bundle"] = json.loads(
            (
                args.baseline_root
                / "full-reference/artifacts/target-calibration-v2.json"
            ).read_text()
        )
        report["trials"]["full-reference"]["coverage"] = json.loads(
            (args.baseline_root / "full-reference/artifacts/coverage.json").read_text()
        )
        report["effects"]["full-reference"] = compare(baseline, baseline)
        report["source_effects"]["full-reference"] = compare(
            source_baseline, source_baseline
        )
        report["effects"]["zero-shot"] = compare(baseline, baseline, mode="zero-shot")
        report["source_effects"]["zero-shot"] = compare(
            source_baseline, source_baseline, mode="zero-shot"
        )
        report["reused_baseline_root"] = str(args.baseline_root.resolve())
        if prior["checkpoint_sha256"] != report["checkpoint_sha256"]:
            raise ValueError("reused baseline checkpoint mismatch")
    for name, policy in INTERVENTIONS.items():
        if args.variants and name not in args.variants:
            continue
        if args.baseline_root and name == "full-reference":
            continue
        if name == "zero-shot":
            continue  # reference-free baseline is evaluated alongside full-reference
        began = time.monotonic()
        print(f"Starting frozen-weight trial: {name}", flush=True)
        root = args.output / name
        root.mkdir()
        os.environ["GRAPHIDS_RUN_DIR"] = str(root.resolve())
        if dm is None:
            dm = CANSnapshotDataModule(
                manifest_path=str(args.manifest),
                datasets=["set_01"],
                cache_root=str(args.cache_root),
                reference_policy=policy,
            )
            dm.setup()
        else:
            dm.reference_policy = (
                policy  # immutable raw contexts; lazy lifecycle cache identity
            )

        net = VehicleGraphMambaJoint.load_from_checkpoint(
            args.checkpoint, map_location="cpu"
        )
        net.reference_policy = policy
        net.calibration_bundles = {}
        net.prepare_from_datamodule(dm)
        if args.baseline_root:
            for mode, payload in report["trials"]["full-reference"]["bundle"].items():
                CalibrationBundle(**payload).validate(
                    weights_digest(net),
                    net.source_manifest_digest,
                    INTERVENTIONS["full-reference"].digest,
                    mode,
                )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.validate(net, datamodule=dm, verbose=False)
        source = ScoreTable(net._validation_scores)
        source.apply_bundles(net.calibration_bundles)
        (root / "source-metrics-v2.json").write_text(
            json.dumps(source.report(net.calibration_bundles), indent=2)
        )
        with (root / "source-scores-v2.jsonl").open("w") as handle:
            for row in source:
                handle.write(json.dumps(row) + "\n")
        # Persist frozen source bundle before target replay.
        (root / "frozen-source-calibration-v2.json").write_text(
            json.dumps(
                {k: v.to_dict() for k, v in net.calibration_bundles.items()}, indent=2
            )
        )
        trainer.test(net, datamodule=dm, verbose=False)
        table = load_scores(root / "artifacts/bin-scores.jsonl")
        if baseline is None:
            baseline, source_baseline = table, source
            report["effects"]["zero-shot"] = compare(table, baseline, mode="zero-shot")
            report["source_effects"]["zero-shot"] = compare(
                source, source_baseline, mode="zero-shot"
            )
        print(f"Completed {name} in {time.monotonic() - began:.1f}s", flush=True)
        report["effects"][name] = compare(table, baseline)
        report["source_effects"][name] = compare(source, source_baseline)
        report["trials"][name] = {
            "wall_seconds": time.monotonic() - began,
            "coverage": json.loads((root / "artifacts/coverage.json").read_text()),
            "target_bundle": json.loads(
                (root / "artifacts/target-calibration-v2.json").read_text()
            ),
            "source": source.report(net.calibration_bundles),
            "strict": json.loads((root / "artifacts/metrics-v2.json").read_text()),
            "target_calibrated": json.loads(
                (root / "artifacts/target-calibrated-metrics-v2.json").read_text()
            ),
            "bundle": {k: v.to_dict() for k, v in net.calibration_bundles.items()},
        }
        (args.output / "diagnostics-v2.json").write_text(json.dumps(report, indent=2))
    if args.last_checkpoint:
        root = args.output / "last-checkpoint"
        root.mkdir()
        os.environ["GRAPHIDS_RUN_DIR"] = str(root.resolve())
        dm.reference_policy = INTERVENTIONS["full-reference"]
        net = VehicleGraphMambaJoint.load_from_checkpoint(
            args.last_checkpoint, map_location="cpu"
        )
        net.prepare_from_datamodule(dm)
        trainer.validate(net, datamodule=dm, verbose=False)
        last_source = ScoreTable(net._validation_scores)
        last_source.apply_bundles(net.calibration_bundles)
        report["last_checkpoint_source"] = last_source.report(net.calibration_bundles)
        report["last_checkpoint_identity"] = {
            "path": str(args.last_checkpoint.resolve()),
            "sha256": hashlib.sha256(args.last_checkpoint.read_bytes()).hexdigest(),
            "same_file_contents_as_selected": hashlib.sha256(
                args.last_checkpoint.read_bytes()
            ).hexdigest()
            == report["checkpoint_sha256"],
        }
        (root / "frozen-source-calibration-v2.json").write_text(
            json.dumps(
                {k: v.to_dict() for k, v in net.calibration_bundles.items()}, indent=2
            )
        )
        (root / "source-metrics-v2.json").write_text(
            json.dumps(report["last_checkpoint_source"], indent=2)
        )
        trainer.test(net, datamodule=dm, verbose=False)
        report["last_checkpoint"] = json.loads(
            (root / "artifacts/metrics-v2.json").read_text()
        )
    (args.output / "diagnostics-v2.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
