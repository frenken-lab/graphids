"""Three sequential, alternating eager/graph pairs at fixed 256-lane geometry."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from graphids.exp.config import load_experiment_yaml
from graphids.exp.trials import run_trial, validate_trial


def select_pairs(pairs):
    gains = []
    for eager, graph in pairs:
        validate_trial(graph, eager)
        if graph["cuda_graph"].get("disabled") or not graph["cuda_graph"].get(
            "replays", 0
        ):
            return {"accepted": False, "reason": "capture did not replay successfully"}
        if graph["optimizer_steps"] != eager["optimizer_steps"]:
            raise ValueError("optimizer update counts differ")
        gains.append(
            graph["end_to_end_scored_events_per_second"]
            / eager["end_to_end_scored_events_per_second"]
            - 1
        )
    median = statistics.median(gains)
    return {
        "accepted": median >= 0.03 and sum(g > 0 for g in gains) >= 2,
        "median_gain": median,
        "paired_gains": gains,
    }


def run_pairs(root):
    root.mkdir(parents=True, exist_ok=True)
    base = load_experiment_yaml("configs/experiments/comparisons/temporal_set_01_segmented_lanes256_scaling_diagnostic.yml")
    base["config"]["data"].update(
        num_workers=2, prebuild_batches=2, persistent_workers=True
    )
    base["config"]["model"]["scan"]["ssm_kernel"] = "torch"
    pairs = []
    for pair in range(3):
        results = {}
        for enabled in (False, True) if pair % 2 == 0 else (True, False):
            name = f"pair{pair + 1}_{'graph' if enabled else 'eager'}"
            config = json.loads(json.dumps(base))
            config["experiment_name"] = "temporal_cuda_" + name
            config["config"]["model"]["scan"]["cuda_graph"] = enabled
            results[enabled] = run_trial(root, name, config=config, section_limit=2048, reference=pairs[0][0] if pairs else None)
            (root / "progress.json").write_text(
                json.dumps({"latest": name, "result": results[enabled]}, indent=2)
                + "\n"
            )
        pairs.append((results[False], results[True]))
    report = {
        "selection": select_pairs(pairs),
        "pairs": pairs,
        "note": "Capture setup included in fit timing. Full epochs require acceptance and complete CUDA correctness checks.",
    }
    (root / "pairs.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_pairs(args.root)["selection"]), flush=True)


if __name__ == "__main__":
    main()
