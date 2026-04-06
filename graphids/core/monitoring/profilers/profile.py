"""Profile training — run a short profiled fit with PyTorchProfiler overlay.

Usage:
    python -m graphids profile
    python -m graphids profile supervised small set_01
    python -m graphids profile fusion large set_01 --fusion-method bandit
"""

from __future__ import annotations

import argparse

from graphids.core.train_entrypoint import run_training_from_spec

from graphids.config.topology import STAGE_FAMILY_MAP
from graphids.orchestrate.contracts import TrainingContract, TrainingSpec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile a training stage")
    parser.add_argument("stage", nargs="?", default="autoencoder")
    parser.add_argument("scale", nargs="?", default="small")
    parser.add_argument("dataset", nargs="?", default="hcrl_ch")
    parser.add_argument("--model-family", dest="model_family")
    parser.add_argument("--fusion-method", default="dqn")
    args = parser.parse_args(argv)

    scale = TrainingContract.normalize_scale(args.scale)
    family = args.model_family or STAGE_FAMILY_MAP.get(args.stage)
    if not family:
        raise ValueError(f"Cannot infer model family for stage '{args.stage}'. Use --model-family.")

    tla: dict = {
        "dataset": args.dataset,
        "seed": 42,
        "run_dir": "",
        "scale": scale,
        "trainer_overrides": {"trainer.profiler": "simple"},
        "stage_overrides": {},
    }
    if args.stage == "fusion":
        tla["fusion_method"] = args.fusion_method

    spec = TrainingSpec(
        stage=args.stage,
        model_family=family,
        scale=scale,
        dataset=args.dataset,
        seed=42,
        run_dir="",
        jsonnet_path=TrainingContract.resolve_jsonnet_path(args.stage),
        jsonnet_tla=tla,
    )

    run_training_from_spec(spec)
