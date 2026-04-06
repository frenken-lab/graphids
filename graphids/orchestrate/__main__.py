"""Dagster orchestration CLI: python -m graphids.orchestrate.

The legacy validate commands were removed in the config reorg. Use
``dg launch`` / ``dg list defs`` for dagster, and rely on the Pydantic
validation gates in the resolver.
"""

from __future__ import annotations

import sys


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else None

    if cmd:
        print("No orchestrate subcommands remain. Use dg CLI or graphids commands.", file=sys.stderr)
        sys.exit(1)
    print("Usage: python -m graphids.orchestrate", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
