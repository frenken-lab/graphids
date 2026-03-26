"""Project-wide constants and pipeline topology loader.

No imports from other config submodules — this is the leaf dependency.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = CONFIG_DIR / "datasets.yaml"

# ---------------------------------------------------------------------------
# Preprocessing constants
# ---------------------------------------------------------------------------
PREPROCESSING_VERSION = "7.0.0"
MAX_DATA_BYTES = 8
EXCLUDED_ATTACK_TYPES = ["suppress", "masquerade"]


def compute_preprocessing_hash() -> str:
    """Content-addressable hash of preprocessing parameters."""
    import hashlib

    # Lazy import to avoid circular dependency (features.py → constants.py)
    from graphids.core.preprocessing.features import N_EDGE_FEATURES, N_NODE_FEATURES

    components = [PREPROCESSING_VERSION, str(N_NODE_FEATURES), str(N_EDGE_FEATURES), "100", "100", "0.8"]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Pipeline topology (derived from pipeline.yaml at import time)
# ---------------------------------------------------------------------------
_pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())

STAGES: dict[str, tuple[str, str, str]] = {
    name: (s["learning_type"], s["model"], s["mode"])
    for name, s in _pipeline["stages"].items()
}
STAGE_MODEL_MAP: dict[str, str] = {k: v[1] for k, v in STAGES.items()}
STAGE_DEPENDENCIES: dict[str, list[tuple[str, str]]] = {
    name: [(d["model"], d["stage"]) for d in s.get("depends_on", [])]
    for name, s in _pipeline["stages"].items()
    if s.get("depends_on")
}
VALID_MODEL_TYPES: frozenset[str] = frozenset(_pipeline["models"])
VALID_SCALES: frozenset[str] = frozenset(_pipeline["scales"])
# Full pipeline dict for orchestration (identity_keys, default_stages).
PIPELINE_YAML: dict = _pipeline
