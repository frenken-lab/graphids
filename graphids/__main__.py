"""CLI entry point: python -m graphids <subcommand>

Training:
  fit / test / validate / predict  — Lightning Trainer methods on stage configs

Analysis:
  analyze                          — generate analysis artifacts from checkpoints

Data:
  rebuild-caches                   — rebuild preprocessed graph caches
  stage-data                       — NFS -> scratch -> TMPDIR staging
  extract-fusion-states            — extract VGAE+GAT latent states for fusion

Orchestration:
  from-spec --phase {train,test,analyze}  — run stage from canonical spec
  pipeline-status                  — aggregated status from DuckDB catalog
  rebuild-catalog                  — rebuild DuckDB from run_record.json sidecars

SLURM:
  submit-profile <job>             — print resource profile for submit.sh
  probe-budget                     — hardware cost model measurement

Dagster (separate entry point):
  dg launch --assets ...           — materialize assets
"""

from __future__ import annotations

from graphids.log import configure_logging

configure_logging()

# Register command modules (each decorates app with @app.command)
import graphids.cli._analysis  # noqa: E402, F401
import graphids.cli._data  # noqa: E402, F401
import graphids.cli._orchestrate  # noqa: E402, F401
import graphids.cli._slurm  # noqa: E402, F401
import graphids.cli._training  # noqa: E402, F401
from graphids.cli.app import app  # noqa: E402


def main() -> None:
    app()


if __name__ == "__main__":
    main()
