"""Dagster orchestration CLI: python -m graphids.orchestrate <command>

Commands:
  validate   — validate all recipe config chains parse correctly
"""

from __future__ import annotations

import sys


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else None

    if cmd == "validate":
        from graphids.orchestrate.validate import main as validate_main
        validate_main(sys.argv[2:])
    else:
        print("Usage: python -m graphids.orchestrate <command>", file=sys.stderr)
        print("Commands: validate", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
