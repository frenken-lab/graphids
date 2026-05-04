"""CLI entry point: ``python -m graphids <subcommand>``.

Surface (the four-step chassis):
  Plans:     run     render+validate plan.jsonnet → JSON array
  Execution: exec    run one row in-process
  SLURM:     submit  submit one row via Parsl SlurmProvider; prints jid
"""

from __future__ import annotations

import graphids.cli.commands  # noqa: F401  -- registers run/exec/submit on `app`
from graphids.cli.app import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
