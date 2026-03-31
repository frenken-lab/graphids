"""CLI entry point: python -m graphids <subcommand>

Subcommands:
  fit|test|validate|predict  — LightningCLI training/evaluation
  analyze                    — generate analysis artifacts from checkpoints
  profile <job_ids>          — sacct resource profiler
  run                        — dagster asset materialization
  validate-recipe            — verify all recipe config chains parse
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Process-level setup (must run before any torch import)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None

    if cmd == "analyze":
        from jsonargparse import ArgumentParser
        from graphids.core.artifacts import Analyzer
        parser = ArgumentParser()
        parser.add_class_arguments(Analyzer, "analyzer")
        cfg = parser.parse_args(sys.argv[2:])
        parser.instantiate_classes(cfg).analyzer.run()

    elif cmd == "profile":
        from graphids.orchestrate.profiler import main as profile_main
        profile_main(sys.argv[2:])

    elif cmd == "run":
        from graphids.orchestrate.run import run_orchestrate
        run_orchestrate(sys.argv[2:])

    elif cmd == "validate-recipe":
        from graphids.orchestrate.validate import validate_recipe
        validate_recipe(sys.argv[2:])

    else:
        from graphids.cli import CLI_KWARGS, GraphIDSCLI
        GraphIDSCLI(**CLI_KWARGS)
