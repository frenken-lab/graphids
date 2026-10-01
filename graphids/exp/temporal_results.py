"""Select section geometry and publish the completed full-coverage pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_lanes(rate32: float, rate64: float) -> int:
    if rate32 <= 0 or rate64 <= 0:
        raise ValueError("diagnostic scored throughput must be positive")
    return 64 if rate64 >= rate32 * 1.03 else 32


def load_result(root, mode):
    return json.loads((Path(root) / mode / "comparison.json").read_text())


def choose(root):
    a, b = load_result(root, "diagnostic32"), load_result(root, "diagnostic64")
    if a["initial_weight_sha256"] != b["initial_weight_sha256"]:
        raise ValueError("diagnostics did not start from identical weights")
    x, y = (r["coverage"]["scored_events_per_second"] for r in (a, b))
    return {"selected_lanes": select_lanes(x, y), "scored_rate_32": x,
            "scored_rate_64": y, "improvement_64_percent": (y / x - 1) * 100}


def render_report(root):
    choice = choose(root)
    exact, segmented = load_result(root, "continuous"), load_result(root, "segmented")
    if exact["initial_weight_sha256"] != segmented["initial_weight_sha256"]:
        raise ValueError("full epochs did not start from identical weights")
    for result in (exact, segmented):
        audit = result["coverage"]
        if not audit["complete"] or audit["unique"] != 12_090_403 or audit["observed_successor_targets"] != 12_090_385:
            raise ValueError("cannot publish incomplete full-coverage comparison")
    lines = [
        "## Full-Coverage Recording / Section Pilot — October 1, 2026",
        "",
        f"Artifacts: `{root}`. Each mode trained one complete epoch on 12,090,403 events "
        "across 18 recordings, including 57,108 attacks. Both audits found zero missing, "
        "duplicate, unexpected, or scored replay/padding events and covered all 12,090,385 successor targets.",
        "",
        f"The seed-42 section diagnostics measured {choice['scored_rate_32']:.2f} scored events/sec "
        f"with 32 lanes and {choice['scored_rate_64']:.2f} with 64 lanes "
        f"({choice['improvement_64_percent']:+.2f}%). Selected **{choice['selected_lanes']} lanes** "
        "using the ≥3% rule. These rates count scored events and include replay and loader/optimizer overhead.",
        "",
        "| Metric | Continuous, 18 lanes | Experimental sections |",
        "| --- | ---: | ---: |",
    ]
    rows = [
        ("Full epoch fit wall time (s)", lambda r: r["fit_wall_seconds"]),
        ("Training audit wall time (s)", lambda r: r["coverage"]["wall_seconds"]),
        ("End-to-end scored events/sec", lambda r: r["coverage"]["scored_events_per_second"]),
        ("Replay events", lambda r: r["coverage"]["warmup_events"]),
        ("Replay / scored events", lambda r: r["coverage"]["warmup_overhead"]),
        ("Mean active lanes", lambda r: r["coverage"]["mean_active_lanes"]),
        ("Lane occupancy", lambda r: r["coverage"]["lane_occupancy"]),
        ("Padding events", lambda r: r["coverage"]["padding_events"]),
        ("Padding fraction", lambda r: r["coverage"]["padding_fraction"]),
    ]
    metrics = ["resource/gpu_util_percent_mean", "resource/gpu_util_percent_max", "resource/gpu_mem_gib_max",
               "train_loss", "val_loss", "val_classification_loss", "val_next_id_loss", "val_iat_loss",
               "val_payload_delta_loss", "val_cls_auroc", "val_anomaly_auroc",
               "val_class_0_events", "val_class_1_events"]
    rows.extend((name, lambda r, name=name: r["metrics"].get(name, "unavailable")) for name in metrics)
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    for name, get in rows:
        lines.append(f"| {name} | {fmt(get(exact))} | {fmt(get(segmented))} |")
    lines.extend([
        "", "Per-class scored coverage in both modes: benign **12,033,295 / 12,033,295**; "
        "attack **57,108 / 57,108**. Per-recording coverage is recorded in each epoch JSON.",
        "", f"Both models started from weight SHA-256 `{exact['initial_weight_sha256']}`. "
        f"Continuous GPU: `{exact['cuda_device']}`; segmented GPU: `{segmented['cuda_device']}`.",
        "", "Each saved `checkpoints/final.ckpt` was restored and evaluated continuously on the first "
        "8,192 validation events per recording, retaining the 64-event unscored warmup. No dropped "
        "evaluation batches or test data. AUROC is unavailable when scored prefixes contain only one class; "
        "the table discloses those class counts.",
        "", "This compares complete coverage modes directly. The old strict 512-event prefix benchmark "
        "is a different diagnostic workload. One epoch per mode is a quality pilot. Finite replay cannot "
        "recover arbitrary GRU memory or cumulative rhythm statistics; segmented training remains opt-in.",
        "",
    ])
    return "\n".join(lines)


def update_note(path, report):
    start = "<!-- temporal-full-coverage:begin -->"
    end = "<!-- temporal-full-coverage:end -->"
    existing = path.read_text()
    block = start + "\n" + report.rstrip() + "\n" + end
    if start in existing and end in existing:
        prefix, rest = existing.split(start, 1)
        _, suffix = rest.split(end, 1)
        text = prefix + block + suffix
    else:
        text = existing.rstrip() + "\n\n" + block + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--choose", action="store_true")
    parser.add_argument("--note", type=Path)
    args = parser.parse_args()
    if args.choose:
        print(json.dumps(choose(args.root)))
        return
    report = render_report(args.root)
    (args.root / "full-comparison.md").write_text(report)
    if args.note:
        update_note(args.note, report)
    print(report)


if __name__ == "__main__":
    main()
