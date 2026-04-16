"""CLI entry point: python -m graphids <subcommand>

Training:
  fit / test                       — Trainer methods on jsonnet stage configs

Analysis:
  analyze                          — generate analysis artifacts from checkpoints

Data:
  rebuild-caches                   — rebuild preprocessed graph caches
  extract-fusion-states            — extract VGAE+GAT latent states for fusion

Catalog:
  catalog-query                    — filter rows from the cross-run parquet catalog
  catalog-rebuild                  — backfill catalog rows from existing summary.json files

SLURM:
  submit-profile                   — print resource profile (consumed by submit.sh)
"""

from __future__ import annotations

# Register command modules (each decorates app with @app.command).
# init_providers() + log-level setup runs inside ``app.py``'s @app.callback,
# so this module has no import-time side effects beyond decorator registration.
# ``submit-profile`` lives directly in ``app.py`` (no domain coupling).
import graphids.cli.analysis  # noqa: F401
import graphids.cli.catalog  # noqa: F401
import graphids.cli.data  # noqa: F401
import graphids.cli.training  # noqa: F401
from graphids.cli.app import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
