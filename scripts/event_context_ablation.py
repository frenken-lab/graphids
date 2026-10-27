"""Generate isolated Section 4 ablations without submitting training jobs."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def experiments(base, *, seed=42):
    variants = {
        "baseline": {},
        "no_elapsed": {"use_elapsed_time": False},
        "normalized_bus": {"timing_mode": "normalized"},
        "id_gap": {"use_id_gap": True},
        "payload_delta": {"payload_context": "delta"},
        "payload_xor": {"payload_context": "xor"},
        "graph_age": {"graph_age": True},
    }
    for name, options in variants.items():
        cfg = deepcopy(base)
        cfg["experiment_name"] = f"event_context_{name}_seed{seed}"
        cfg["seed"] = seed
        model = cfg["config"]["model"]
        # Explicitly reset every feature, even when extending an ablated base.
        model.update(
            use_elapsed_time=True,
            timing_mode="raw",
            use_id_gap=False,
            payload_context="none",
            graph_age=False,
            graph_age_scale=1.0,
            timing_min_support=32,
            schema_version=1 if not options else 2,
        )
        model.update(options)
        version = 2 if options.get("use_id_gap") or "payload_context" in options else 1
        representation = dict(kind="can_event", schema_version=version)
        cfg["representation_cfg"] = representation
        cfg["config"]["data"]["representation_cfg"] = deepcopy(representation)
        yield cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    preset = (
        Path(__file__).resolve().parents[1]
        / "configs/experiments/graph_mamba_events_source.yml"
    )
    base = yaml.safe_load(preset.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for cfg in experiments(base, seed=args.seed):
        path = args.output / f"{cfg['experiment_name']}.yml"
        with path.open("x") as output:
            yaml.safe_dump(cfg, output, sort_keys=False)
        print(path)


if __name__ == "__main__":
    main()
