"""Summarize a submitted event suite without loading weights or retuning thresholds.

python scripts/event_ablation_report.py /path/to/suite.json
Only complete, comparable fits produce a complete comparison. Partial learning
curves remain visible, and the reported 0.5 operating point is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def summarize_run(config, run_dir):
    manifest = run_dir / ".graphids/manifest.json"
    status = (
        json.loads(manifest.read_text())["status"] if manifest.exists() else "pending"
    )
    epochs, coverage, population = [], None, None
    for path in sorted((run_dir / "artifacts").glob("val-events-epoch*/index.json")):
        index = json.loads(path.read_text())
        if index.get("prediction_unit") != "observed_frame":
            raise ValueError(f"unexpected prediction unit: {path}")
        if not index.get("coverage") or set(index.get("by_recording", {})) != set(
            index["coverage"]
        ):
            raise ValueError(f"missing recording coverage or metrics: {path}")
        current = index["coverage"]
        metrics = index["metrics"]
        counts = {
            k: metrics[k]
            for k in (
                "classification_frames",
                "classification_positives",
                "classification_negatives",
            )
        }
        if coverage is not None and (current != coverage or counts != population):
            raise ValueError(f"evaluation cohort changed between epochs: {path}")
        coverage, population = current, counts
        ap = metrics.get("classification_frame_ap")
        if ap is None or not math.isfinite(ap):
            raise ValueError(f"missing finite classification AP: {path}")
        epochs.append(
            dict(
                epoch=int(path.parent.name.removeprefix("val-events-epoch")),
                index=str(path),
                metrics=metrics,
                by_recording=index["by_recording"],
            )
        )
    expected = config["config"]["trainer"]["max_epochs"]
    complete = status == "finished" and [e["epoch"] for e in epochs] == list(
        range(expected)
    )
    # Iteration order makes tied AP select the first epoch, as top-k checkpointing does.
    best = (
        max(epochs, key=lambda e: e["metrics"]["classification_frame_ap"])
        if epochs
        else None
    )
    return dict(
        status="complete" if complete else "incomplete",
        run_status=status,
        run_dir=str(run_dir),
        expected_epochs=expected,
        completed_epochs=len(epochs),
        coverage=coverage,
        label_population=population,
        payload_trained=config["config"]["model"].get("payload_weight", 1.0) > 0,
        best=best,
        final=epochs[-1] if epochs else None,
        learning_curve=[dict(epoch=e["epoch"], metrics=e["metrics"]) for e in epochs],
    )


def suite_report(path):
    suite = json.loads(path.read_text())
    root = Path(suite["root"])
    arms, errors = [], []
    cohort = None
    for job in suite["training_jobs"]:
        cfg = yaml.safe_load(Path(job["config"]).read_text())
        name = cfg["experiment_name"]
        run_dir = root / "runs" / cfg["dataset"] / name / name
        arm = dict(name=name, job_id=job["job_id"], config=job["config"])
        try:
            arm.update(summarize_run(cfg, run_dir))
            if arm["coverage"] is not None:
                current = (arm["coverage"], arm["label_population"])
                if cohort is not None and current != cohort:
                    raise ValueError(
                        "evaluation coverage or label population differs across arms"
                    )
                cohort = current
        except (KeyError, ValueError) as exc:
            arm.update(status="invalid", error=str(exc))
            errors.append(f"{name}: {exc}")
        arms.append(arm)
    complete = bool(arms) and all(a["status"] == "complete" for a in arms)
    return dict(
        schema_version=1,
        status="complete" if complete else "incomplete",
        selection="maximum aggregate validation classification AP; earliest epoch on ties",
        interpretation="Exploratory source validation, seed 42. No independent test or alert-budget assessment.",
        threshold="0.5 diagnostic only; no threshold fitted by this report",
        payload="Classification-only arm has untrained payload scores; exclude them from detector comparisons.",
        errors=errors,
        arms=arms,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = suite_report(args.suite)
    output = args.output or args.suite.parent / "comparison-results.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(f"{report['status']}: {output}")
    for arm in report["arms"]:
        best = arm.get("best")
        detail = (
            f"best AP={best['metrics']['classification_frame_ap']:.6f}, epoch={best['epoch'] + 1}"
            if best
            else "no complete validation epoch"
        )
        print(f"{arm['job_id']} {arm['status']}: {detail}")


if __name__ == "__main__":
    main()
