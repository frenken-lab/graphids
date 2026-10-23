"""Generate matched 25-epoch source experiments; no target data or submission.

python scripts/event_ablation.py OUTPUT --positive-weight NEGATIVE_TRAIN/POSITIVE_TRAIN
Use a numeric ratio (46.6608 for the reviewed October 16 training cohort).
Submit generated YAMLs with `gx exp submit PATH`.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def experiments(base, positive_weight, *, classification_only=False):
    """Keep chunk length, optimizer window, seed and epochs fixed across arms."""
    for streams in (1, 4):
        for weighted in (False, True):
            cfg = deepcopy(base)
            cfg["experiment_name"] = (
                f"event_ablation_streams{streams}_weighted{int(weighted)}_seed42"
            )
            cfg["seed"] = 42
            cfg["config"]["model"].update(
                positive_weight=positive_weight if weighted else 1.0,
                max_graph_tokens=8388608,
            )
            cfg["config"]["data"].update(
                chunk_events=16384,
                stream_batch_size=streams,
                effective_batch_events=65536,
                max_microbatch_events=65536,
            )
            cfg["config"]["trainer"].update(max_epochs=25, enable_checkpointing=True)
            yield cfg
    if classification_only:
        control = next(experiments(base, positive_weight))
        control["experiment_name"] += "_classification_only"
        control["config"]["model"]["payload_weight"] = 0.0
        yield control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--positive-weight", type=float, required=True)
    parser.add_argument(
        "--classification-only",
        action="store_true",
        help="Also generate a matched one-stream unweighted control with payload_weight=0",
    )
    args = parser.parse_args()
    if not 0 < args.positive_weight < float("inf"):
        parser.error("positive weight must be finite and positive")
    preset = (
        Path(__file__).resolve().parents[1]
        / "configs/experiments/graph_mamba_events_source.yml"
    )
    base = yaml.safe_load(preset.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for cfg in experiments(
        base, args.positive_weight, classification_only=args.classification_only
    ):
        path = args.output / f"{cfg['experiment_name']}.yml"
        with path.open("x") as output:
            yaml.safe_dump(cfg, output, sort_keys=False)
        print(path)


if __name__ == "__main__":
    main()
