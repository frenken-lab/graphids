"""Profile training — run a short profiled fit with PyTorchProfiler overlay.

Usage:
    python -m graphids profile-training
    python -m graphids profile-training normal small set_01
    python -m graphids profile-training fusion large set_01 --fusion-method bandit
"""

from __future__ import annotations

import argparse

from graphids.config import STAGE_MODEL_MAP
from graphids.core.contracts import TrainingContract, TrainingSpec
from graphids.core.train_entrypoint import run_training_from_spec


def run_profile_training(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile a training stage")
    parser.add_argument("stage", nargs="?", default="autoencoder")
    parser.add_argument("scale", nargs="?", default="small")
    parser.add_argument("dataset", nargs="?", default="hcrl_ch")
    parser.add_argument("--model-family", dest="model_family")
    parser.add_argument("--fusion-method", default="dqn")
    args = parser.parse_args(argv)

    scale = TrainingContract.normalize_scale(args.scale)
    family = args.model_family or STAGE_MODEL_MAP.get(args.stage)
    if not family:
        raise ValueError(f"Cannot infer model family for stage '{args.stage}'. Use --model-family.")

    spec = TrainingSpec(
        stage=args.stage,
        model_family=family,
        scale=scale,
        dataset=args.dataset,
        seed=42,
        run_dir="",
        config_files=TrainingContract.resolve_config_files(
            args.stage,
            scale,
            model_family=family,
            fusion_method=args.fusion_method,
        ),
        runtime_overrides={"trainer.profiler": "simple"},
    )

    run_training_from_spec(spec)


main = run_profile_training
