"""CLI entry point: python -m graphids <subcommand>

Training:
  fit / test / validate / predict  — Trainer methods on jsonnet stage configs

Analysis:
  analyze                          — generate analysis artifacts from checkpoints

Data:
  rebuild-caches                   — rebuild preprocessed graph caches
  extract-fusion-states            — extract VGAE+GAT latent states for fusion

Orchestration:
  pipeline-run                     — run the full 3-stage pipeline in-process

SLURM:
  probe-budget                     — hardware cost model measurement
"""

from __future__ import annotations

# Register command modules (each decorates app with @app.command).
# init_providers() + log-level setup runs inside ``app.py``'s @app.callback,
# so this module has no import-time side effects beyond decorator registration.
import graphids.cli._analysis  # noqa: F401
import graphids.cli._data  # noqa: F401
import graphids.cli._pipeline  # noqa: F401
import graphids.cli._slurm  # noqa: F401
import graphids.cli._training  # noqa: F401
from graphids.cli.app import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
