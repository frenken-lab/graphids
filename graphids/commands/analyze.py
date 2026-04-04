"""Generate analysis artifacts from trained checkpoints.

Usage:
    python -m graphids analyze --config graphids/config/stages/analyze_vgae.yaml \\
        --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa

    # Landscape subcommand:
    python -m graphids analyze landscape vgae hcrl_sa path/to/best.ckpt [--resolution 51]
"""

from __future__ import annotations

import sys

from graphids.config import CONFIG_DIR


def _run_analyze(argv: list[str]) -> None:
    from jsonargparse import ArgumentParser

    from graphids.core.artifacts import Analyzer

    parser = ArgumentParser()
    parser.add_class_arguments(Analyzer, "analyzer")
    cfg = parser.parse_args(argv)
    parser.instantiate_classes(cfg).analyzer.run()


def _run_landscape(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute 2D loss landscape")
    parser.add_argument("model_type", choices=["vgae", "gat", "fusion"])
    parser.add_argument("dataset")
    parser.add_argument("ckpt_path")
    parser.add_argument("--resolution", type=int, default=51)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args(argv)

    config_file = CONFIG_DIR / "stages" / f"analyze_{args.model_type}.yaml"
    if not config_file.exists():
        print(f"No analyze config for model_type={args.model_type}: {config_file}", file=sys.stderr)
        sys.exit(1)

    _run_analyze([
        f"--config={config_file}",
        f"--analyzer.ckpt_path={args.ckpt_path}",
        f"--analyzer.dataset={args.dataset}",
        "--analyzer.embeddings=false",
        "--analyzer.landscape=true",
        f"--analyzer.landscape_resolution={args.resolution}",
        f"--analyzer.landscape_scale={args.scale}",
    ])


def main(argv: list[str]) -> None:
    if argv and argv[0] == "landscape":
        _run_landscape(argv[1:])
    else:
        _run_analyze(argv)
