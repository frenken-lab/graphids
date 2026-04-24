"""CLI entry point: python -m graphids <subcommand>

Training:
  fit / test                       — Trainer methods on jsonnet stage configs

Analysis:
  analyze                          — generate analysis artifacts from checkpoints

Data:
  rebuild-caches                   — rebuild preprocessed graph caches
  extract-fusion-states            — extract VGAE+GAT latent states for fusion

SLURM:
  launch-ablation                  — submit the OFAT ablation DAG (topology
                                     in ``graphids.slurm.dag.OFAT_DAG``)
  (single jobs: use scripts/run <preset.jsonnet> or scripts/run --mode cpu --command "...")
"""

from __future__ import annotations

# Register command modules (each decorates app with @app.command).
# init_providers() + log-level setup runs inside ``app.py``'s @app.callback,
# so this module has no import-time side effects beyond decorator registration.
import graphids.cli.ablation  # noqa: F401
import graphids.cli.analysis  # noqa: F401
import graphids.cli.compare  # noqa: F401
import graphids.cli.data  # noqa: F401
import graphids.cli.training  # noqa: F401
from graphids.cli.app import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
