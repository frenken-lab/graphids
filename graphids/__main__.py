"""CLI entry point: LightningCLI with linked args for DRY config."""

from __future__ import annotations

import torch
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
mp.set_sharing_strategy("file_system")

import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    cache_logger_on_first_use=True,
)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        from jsonargparse import ArgumentParser
        from graphids.core.artifacts import Analyzer

        parser = ArgumentParser(description="Generate analysis artifacts from trained checkpoints")
        parser.add_class_arguments(Analyzer, "analyzer")
        cfg = parser.parse_args(sys.argv[2:])
        analyzer = parser.instantiate_classes(cfg).analyzer
        analyzer.run()
    else:
        from graphids.cli import GraphIDSCLI, CLI_KWARGS

        GraphIDSCLI(**CLI_KWARGS)
