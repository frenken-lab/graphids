"""CLI entry point: python -m graphids <subcommand>

Surface (the four-step chassis):
  Plans:     run     render+validate plan.jsonnet → JSON array
  Execution: exec    run one row in-process
  SLURM:     submit  submit one row via Parsl SlurmProvider; prints jid

`analyze`, `rebuild-caches`, `extract-fusion-states`, `push-hf`, etc. are
separate subsystems not on this chassis. They're outside this rebuild's scope
and stay un-registered until they're ported.
"""

from __future__ import annotations

import graphids.cli.exec  # noqa: F401
import graphids.cli.run  # noqa: F401
import graphids.cli.submit  # noqa: F401
from graphids.cli.app import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
