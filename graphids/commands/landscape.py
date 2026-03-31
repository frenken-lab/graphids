"""Loss landscape computation — convenience wrapper around Analyzer.

Usage:
    python -m graphids landscape vgae hcrl_sa /path/to/best.ckpt
    python -m graphids landscape gat hcrl_sa /path/to/best.ckpt --resolution 101
"""

from __future__ import annotations

import argparse
import sys

from graphids.config import CONFIG_DIR


def run_landscape(argv: list[str] | None = None) -> None:
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

    from jsonargparse import ArgumentParser as JAParser

    from graphids.core.artifacts import Analyzer

    jp = JAParser()
    jp.add_class_arguments(Analyzer, "analyzer")
    cfg = jp.parse_args([
        f"--config={config_file}",
        f"--analyzer.ckpt_path={args.ckpt_path}",
        f"--analyzer.dataset={args.dataset}",
        "--analyzer.embeddings=false",
        "--analyzer.landscape=true",
        f"--analyzer.landscape_resolution={args.resolution}",
        f"--analyzer.landscape_scale={args.scale}",
    ])
    jp.instantiate_classes(cfg).analyzer.run()


main = run_landscape
