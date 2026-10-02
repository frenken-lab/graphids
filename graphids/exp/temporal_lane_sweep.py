"""Bounded same-section lane scaling on one GPU; no change to production geometry."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from graphids.exp.temporal_results import load_result

LANES = (64, 128, 256, 512, 1024)


def validate_trial(result, reference=None):
    coverage = result["coverage"]
    for key in ("duplicate", "unexpected", "incorrectly_scored_warmup_or_padding",
                "duplicate_successor_targets", "unexpected_successor_targets"):
        if coverage[key]:
            raise ValueError(f"invalid sweep coverage: {key}")
    if coverage["unique"] != result["planned_scored_events"]:
        raise ValueError("trial missed scored core events")
    if coverage["observed_successor_targets"] != result["planned_successor_targets"]:
        raise ValueError("trial missed eligible successor targets")
    if coverage["warmup_events"] != result["planned_warmup_events"]:
        raise ValueError("trial missed replay events")
    if reference is not None:
        for key in ("initial_weight_sha256", "section_queue_sha256", "planned_scored_events", "planned_warmup_events", "planned_successor_targets"):
            if result[key] != reference[key]:
                raise ValueError(f"sweep trials differ in {key}")
        if result["cuda_device"] != reference["cuda_device"]:
            raise ValueError("sweep trials must use the same GPU model")


def should_try_1024(rate256, rate512):
    return rate512 >= rate256 * 1.03


def publish(root, results, note=None, skipped_1024=False):
    lines = ["## Segmented Lane Scaling — October 1, 2026", "",
             f"Artifacts: `{root}`. Each trial processes the same first 2,048 seed-42 "
             "sections, with 32-event chunks, 2,048-event cores, and up to 256 replay events. "
             "Trials start from identical weights and run sequentially on one GPU allocation.", "",
             "| Lanes | Scored events/sec | Gain vs previous | GPU mean / peak | NVML VRAM peak (MB) | Torch allocated / reserved peak (MiB) | Fit wall (s) | Padding | Val loss |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    previous = None
    selected = 64
    selected_rate = results[64]["coverage"]["scored_events_per_second"]
    for lanes, result in results.items():
        rate = result["coverage"]["scored_events_per_second"]
        gain = "—" if previous is None else f"{(rate / previous - 1) * 100:+.1f}%"
        metrics, cuda = result["metrics"], result["cuda_peak_memory"]
        if rate >= selected_rate * 1.03:
            selected, selected_rate = lanes, rate
        gpu = f"{metrics.get('resource/gpu_util_percent_mean', 0):.1f}% / {metrics.get('resource/gpu_util_percent_max', 0):.1f}%"
        memory = metrics.get("resource/gpu_mem_gib_max", 0) * 1024
        lines.append(f"| {lanes} | {rate:,.1f} | {gain} | {gpu} | {memory:.1f} | "
                     f"{cuda['allocated_mib']:.1f} / {cuda['reserved_mib']:.1f} | "
                     f"{result['fit_wall_seconds']:.1f} | {result['coverage']['padding_fraction']:.2%} | "
                     f"{metrics['val_loss']:.4f} |")
        previous = rate
    if skipped_1024:
        lines.extend(["", "1,024 lanes skipped: 512 lanes did not improve scored throughput over 256 by at least 3%."])
    reference = next(iter(results.values()))
    lines.extend(["", f"Each trial scored **{reference['planned_scored_events']:,} unique events** and processed "
                  f"**{reference['planned_warmup_events']:,} replay events**. No duplicated, unexpected, "
                  "scored replay/padding, or omitted planned core events were observed.", "",
                  f"Performance candidate under the ≥3% rule: **{selected} lanes**. This is a throughput "
                  "candidate; production settings remain unchanged. Increasing lanes reduces optimizer "
                  "updates per scored workload. Validation uses the same continuous 8,192-event prefixes "
                  "with 64-event scoring warmup, containing 146,304 benign scored events and zero attacks. "
                  "Both AUROCs are unavailable, so this sweep cannot establish attack generalization.", "",
                  f"GPU: `{reference['cuda_device']}`. Initial weights: `{reference['initial_weight_sha256']}`. "
                  f"Section queue: `{reference['section_queue_sha256']}`.", ""])
    report = "\n".join(lines)
    summary = {"selected_performance_lanes": selected, "skipped_1024": skipped_1024,
               "results": {str(k): v for k, v in results.items()}}
    (root / "sweep.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "sweep.md").write_text(report)
    if note is not None:
        # Append outside the separately managed full-epoch report section.
        with note.open("a") as handle:
            handle.write("\n" + report)
    print(report, flush=True)
    return summary


def run_sweep(root, config_dir, note=None):
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    skipped = False
    for lanes in LANES:
        if lanes == 1024 and not should_try_1024(
            results[256]["coverage"]["scored_events_per_second"],
            results[512]["coverage"]["scored_events_per_second"],
        ):
            skipped = True
            break
        config = config_dir / f"temporal_set_01_segmented_lanes{lanes}_scaling_diagnostic.yml"
        out = root / f"lanes{lanes}"
        # Fresh process per geometry avoids CUDA allocator and telemetry carryover.
        subprocess.run([sys.executable, "-m", "graphids.exp.temporal_comparison", str(config),
                        "--output", str(out), "--section-limit", "2048"], check=True)
        result = load_result(root, f"lanes{lanes}")
        validate_trial(result, results.get(64))
        results[lanes] = result
        print(f"sweep_trial_complete lanes={lanes} scored_rate={result['coverage']['scored_events_per_second']}", flush=True)
    return publish(root, results, note, skipped_1024=skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/experiments/comparisons"))
    parser.add_argument("--note", type=Path)
    args = parser.parse_args()
    run_sweep(args.root, args.config_dir, args.note)


if __name__ == "__main__":
    main()
