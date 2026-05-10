"""CLI entry point: ``python -m graphids <subcommand>``.

Current surface:
  Experiment: exp    inspect manifests / status for the new experiment seam
"""

from __future__ import annotations

import graphids.cli.exp  # noqa: F401  -- registers new experiment-manifest surface
from graphids.cli.app import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
