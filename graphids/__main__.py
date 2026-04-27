"""CLI entry point: python -m graphids <subcommand>

Training:
  fit / test                       — Trainer methods on jsonnet stage configs

Analysis:
  analyze                          — generate analysis artifacts from checkpoints

Data:
  rebuild-caches                   — rebuild preprocessed graph caches
  extract-fusion-states            — extract VGAE+GAT latent states for fusion

SLURM:
  submit                           — submit one SLURM job (preset.jsonnet or --command)

Plan workflow:
  run                              — submit a plan via submitit, deps via afterok
  status                           — query MLflow per plan node
"""

from __future__ import annotations

# Register command modules (each decorates app with @app.command).
# init_providers() + log-level setup runs inside ``app.py``'s @app.callback,
# so this module has no import-time side effects beyond decorator registration.
import graphids.cli.analysis  # noqa: F401
import graphids.cli.compare  # noqa: F401
import graphids.cli.data  # noqa: F401
import graphids.cli.run  # noqa: F401
import graphids.cli.submit  # noqa: F401
import graphids.cli.training  # noqa: F401
from graphids.cli.app import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
