"""Print SLURM resource profile for scripts/submit.sh.

Usage:
    python -m graphids submit-profile tests
    python -m graphids submit-profile landscape

Output (space-separated, consumed by submit.sh via `read`):
    partition cpus mem time signal mode command...
"""

from __future__ import annotations

import sys

from graphids.config import CONFIG_DIR
from graphids.config.yaml_utils import read_yaml


_SUBMIT_PROFILES_PATH = CONFIG_DIR / "resources" / "submit_profiles.yaml"


def main(argv: list[str]) -> None:
    profiles = read_yaml(_SUBMIT_PROFILES_PATH).get("submit_profiles", {})

    if not argv:
        print("Available profiles:", ", ".join(sorted(profiles)), file=sys.stderr)
        sys.exit(1)

    job = argv[0]

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
        p.get("signal", "") or "NONE",
        p["mode"],
        p["command"],
    ]
    print(" ".join(parts))
