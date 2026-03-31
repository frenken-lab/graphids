"""Generate analysis artifacts from trained checkpoints.

Usage:
    python -m graphids analyze --config graphids/config/stages/analyze_vgae.yaml \
        --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa
"""

from __future__ import annotations


def main(argv: list[str]) -> None:
    from jsonargparse import ArgumentParser

    from graphids.core.artifacts import Analyzer

    parser = ArgumentParser()
    parser.add_class_arguments(Analyzer, "analyzer")
    cfg = parser.parse_args(argv)
    parser.instantiate_classes(cfg).analyzer.run()
