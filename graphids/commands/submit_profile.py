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


_SUBMIT_PROFILES_PATH = CONFIG_DIR / "resources" / "submit_profiles.yaml"


def _load_submit_profiles() -> dict:
    if not _SUBMIT_PROFILES_PATH.exists():
        raise FileNotFoundError(
            f"Missing submit profiles config: {_SUBMIT_PROFILES_PATH}."
        )
    return yaml.safe_load(_SUBMIT_PROFILES_PATH.read_text()) or {}


def main(argv: list[str]) -> None:
    resources = _load_submit_profiles()
    profiles = resources.get("submit_profiles", {})

    if not argv:
        print("Available profiles:", ", ".join(sorted(profiles)), file=sys.stderr)
        sys.exit(1)

    job = argv[0]

    if job not in profiles:
        print(f"Unknown profile: {job}", file=sys.stderr)
        print("Available:", ", ".join(sorted(profiles)), file=sys.stderr)
        sys.exit(1)

    p = profiles[job]

    # Validate SLURM fields through ResourceSpec (catches bad time formats, etc.)
    from graphids.slurm import ResourceSpec
    ResourceSpec(
        partition=p["partition"],
        time=p["time"],
        mem=p["mem"],
        cpus_per_task=p["cpus"],
        num_workers=0,
    )

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
