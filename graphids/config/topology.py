"""Pipeline topology and config tree validation."""

from __future__ import annotations

import json
from typing import Any

from .constants import (
    CONFIG_DIR,
    PROJECT_ROOT,
    VALID_FUSION_METHODS,
    VALID_MODEL_FAMILIES,
    VALID_SCALES,
)

# Post Phase 1: jsonnet sources live at <repo>/configs/, not under graphids/
_CONFIGS_DIR = PROJECT_ROOT / "configs"

_topology = json.loads((CONFIG_DIR / "matrix" / "topology.json").read_text())
_STAGE_DEFS: dict[str, dict[str, Any]] = _topology["stages"]

STAGES: dict[str, tuple[str, str, str]] = {
    name: (stage["learning_type"], stage["family"], stage["mode"])
    for name, stage in _STAGE_DEFS.items()
}

PIPELINE_TOPOLOGY: dict[str, Any] = {
    "families": list(VALID_MODEL_FAMILIES),
    "fusion_methods": list(VALID_FUSION_METHODS),
    "scales": list(VALID_SCALES),
    "stages": _STAGE_DEFS,
    "default_stages": _topology.get("default_stages", []),
    "ckpt_stages": _topology.get("ckpt_stages", {}),
}

STAGE_FAMILY_MAP: dict[str, str] = {k: v[1] for k, v in STAGES.items()}
STAGE_DEPENDENCIES: dict[str, list[tuple[str, str]]] = {
    name: [(d["family"], d["stage"]) for d in s.get("depends_on", [])]
    for name, s in _STAGE_DEFS.items()
    if s.get("depends_on")
}

# Cross-validate the jsonnet tree against the topology declared above.
# Each model family must have a libsonnet file; each fusion method must
# have a method libsonnet; every trainable stage must have a stage
# jsonnet. Missing files fail fast at package import time with an
# actionable path — no silent fallbacks.
for _family in VALID_MODEL_FAMILIES:
    if _family == "fusion":
        continue  # fusion uses configs/fusion.libsonnet, not configs/models/
    _lib = _CONFIGS_DIR / "models" / f"{_family}.libsonnet"
    if not _lib.exists():
        raise FileNotFoundError(f"Missing model libsonnet: {_lib}")

_fusion_lib = _CONFIGS_DIR / "fusion.libsonnet"
if not _fusion_lib.exists():
    raise FileNotFoundError(f"Missing fusion dispatch libsonnet: {_fusion_lib}")
_fusion_base = _CONFIGS_DIR / "fusion" / "base.libsonnet"
if not _fusion_base.exists():
    raise FileNotFoundError(f"Missing fusion base libsonnet: {_fusion_base}")
for _method in VALID_FUSION_METHODS:
    _method_file = _CONFIGS_DIR / "fusion" / "methods" / f"{_method}.libsonnet"
    if not _method_file.exists():
        raise FileNotFoundError(f"Missing fusion method libsonnet: {_method_file}")

for _stage in _STAGE_DEFS:
    _stage_file = _CONFIGS_DIR / "stages" / f"{_stage}.jsonnet"
    if not _stage_file.exists():
        raise FileNotFoundError(f"Missing stage jsonnet: {_stage_file}")

    _job_profiles = PROJECT_ROOT / "configs" / "resources" / "job_profiles.json"
if not _job_profiles.exists():
    raise FileNotFoundError(f"Missing job profiles: {_job_profiles}")
else:
    _profiles = json.loads(_job_profiles.read_text())
    for _family in VALID_MODEL_FAMILIES:
        if _family not in _profiles:
            raise FileNotFoundError(f"Missing resource profile for '{_family}' in {_job_profiles}")
