"""CLI entry point: python -m graphids <subcommand>

Subcommands (auto-discovered from graphids/commands/):
  fit|test|validate|predict  — LightningCLI training/evaluation
  analyze                    — generate analysis artifacts from checkpoints
  landscape                  — compute 2D loss landscape
  profile                    — sacct resource profiler
  profile-training           — profiled training run (PyTorchProfiler)
  rebuild-caches             — rebuild preprocessed graph caches
  stage-data                 — stage data from NFS to scratch/TMPDIR
  submit-profile             — print SLURM resource profile for submit.sh
  test-preprocessing         — validate preprocessing pipeline

Dagster (separate entry point):
  python -m graphids.orchestrate validate  — validate config chains
  dg launch --assets ...                   — materialize assets
"""

from __future__ import annotations

import importlib
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
# Dispatch: command name → graphids.commands.<module_name>.main(argv)
# Convention: module name = command name with - replaced by _
# Fallback: LightningCLI handles fit/test/validate/predict
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    module_name = (cmd or "").replace("-", "_")

    try:
        mod = importlib.import_module(f"graphids.commands.{module_name}")
    except (ModuleNotFoundError, ValueError):
        mod = None

    if mod and hasattr(mod, "main"):
        mod.main(sys.argv[2:])
    else:
        from graphids.cli import CLI_KWARGS, GraphIDSCLI
        GraphIDSCLI(**CLI_KWARGS)
