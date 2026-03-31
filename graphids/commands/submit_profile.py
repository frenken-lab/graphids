"""Print SLURM resource profile for scripts/submit.sh.

Usage:
    python -m graphids submit-profile tests
    python -m graphids submit-profile landscape

Output (space-separated, consumed by submit.sh via `read`):
    partition cpus mem time signal mode command...
"""

from __future__ import annotations

import sys

import yaml

from graphids.config import CONFIG_DIR


def main(argv: list[str]) -> None:
    if not argv:
        _resources = yaml.safe_load((CONFIG_DIR / "resources.yaml").read_text())
        profiles = _resources["submit_profiles"]
        print("Available profiles:", ", ".join(sorted(profiles)), file=sys.stderr)
        sys.exit(1)

    job = argv[0]
    resources = yaml.safe_load((CONFIG_DIR / "resources.yaml").read_text())
    profiles = resources["submit_profiles"]

    if job not in profiles:
        print(f"Unknown profile: {job}", file=sys.stderr)
        print("Available:", ", ".join(sorted(profiles)), file=sys.stderr)
        sys.exit(1)

    p = profiles[job]
    parts = [
        p["partition"],
        str(p["cpus"]),
        p["mem"],
        p["time"],
        p.get("signal", ""),
        p["mode"],
        p["command"],
    ]
    print(" ".join(parts))
