"""Print SLURM resource profile for scripts/submit.sh.

Usage:
    python -m graphids submit-profile tests
    python -m graphids submit-profile landscape

Output (space-separated, consumed by submit.sh via `read`):
    partition cpus mem time signal mode gres command...
"""

from __future__ import annotations

import socket
import sys

from graphids.config import CONFIG_DIR
from graphids.config.yaml_utils import read_yaml

_SUBMIT_PROFILES_PATH = CONFIG_DIR / "resources" / "submit_profiles.yaml"
_CLUSTERS_PATH = CONFIG_DIR / "resources" / "clusters.yaml"


def _detect_cluster() -> str:
    """Detect cluster from hostname using clusters.yaml mapping."""
    clusters = read_yaml(_CLUSTERS_PATH)["clusters"]
    hostname = socket.gethostname().lower()
    for cluster, prefixes in clusters["detect_by_hostname"].items():
        if any(hostname.startswith(p) for p in prefixes):
            return cluster
    return clusters["default"]


def _resolve_gres(mode: str) -> str:
    """Look up gres from clusters.yaml execution_modes for the current cluster."""
    if not mode.startswith("gpu"):
        return "NONE"
    cluster = _detect_cluster()
    clusters = read_yaml(_CLUSTERS_PATH)["clusters"]
    exec_modes = clusters.get("execution_modes", {}).get(cluster, {})
    # mode "gpu" maps to "gpu_train" execution mode
    gpu_mode = exec_modes.get("gpu_train", {})
    return gpu_mode.get("gres", "NONE")


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
    gres = _resolve_gres(p["mode"])

    parts = [
        p["partition"],
        str(p["cpus"]),
        p["mem"],
        p["time"],
        p.get("signal", "") or "NONE",
        p["mode"],
        gres,
        p["command"],
    ]
    print(" ".join(parts))
