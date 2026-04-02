"""Pipeline topology and config tree validation."""

from __future__ import annotations

from typing import Any

from .base import CONFIG_DIR
from .yaml_utils import read_yaml

_axes = read_yaml(CONFIG_DIR / "matrix" / "axes.yaml").get("axes", {})

VALID_SCALES: frozenset[str] = frozenset(_axes.get("scales", ["small", "large"]))
VALID_FUSION_METHODS: frozenset[str] = frozenset(_axes.get("fusion_methods", []))
_VALID_MODEL_FAMILIES = [m for m in _axes.get("model_families", []) if m != "fusion"]
VALID_MODEL_TYPES: frozenset[str] = frozenset(_VALID_MODEL_FAMILIES)

STAGES: dict[str, tuple[str, str, str]] = {
    "autoencoder": ("unsupervised", "vgae", "gpu_train"),
    "curriculum": ("supervised", "gat", "gpu_train"),
    "normal": ("supervised", "gat", "gpu_train"),
    "fusion": ("rl_fusion", "fusion", "gpu_train"),
    "temporal": ("temporal", "temporal", "gpu_train"),
}

_STAGE_DEFS: dict[str, dict[str, Any]] = {
    "autoencoder": {
        "learning_type": "unsupervised",
        "model": "vgae",
        "mode": "gpu_train",
        "depends_on": [],
        "identity_keys": ["scale", "conv_type", "variational", "model_type"],
        "model_keys": ["conv_type", "variational"],  # model_type: identity only, set by YAML
    },
    "curriculum": {
        "learning_type": "supervised",
        "model": "gat",
        "mode": "gpu_train",
        "depends_on": [{"model": "vgae", "stage": "autoencoder"}],
        "identity_keys": ["scale", "conv_type", "loss_fn", "variational", "model_type"],
        "model_keys": ["conv_type", "loss_fn"],  # variational/model_type: identity only
    },
    "normal": {
        "learning_type": "supervised",
        "model": "gat",
        "mode": "gpu_train",
        "depends_on": [],
        "identity_keys": ["scale", "conv_type", "loss_fn"],
    },
    "fusion": {
        "learning_type": "rl_fusion",
        "model": "fusion",
        "mode": "gpu_train",
        "depends_on": [
            {"model": "vgae", "stage": "autoencoder"},
            {"model": "gat", "stage": "curriculum"},
            {"model": "gat", "stage": "normal"},
        ],
        "identity_keys": ["scale", "gat_stage", "loss_fn", "method", "conv_type", "variational"],
        "model_keys": [],
    },
    "temporal": {
        "learning_type": "temporal",
        "model": "temporal",
        "mode": "gpu_train",
        "depends_on": [{"model": "gat", "stage": "curriculum"}],
        "identity_keys": ["scale", "gat_stage", "loss_fn"],
    },
}

PIPELINE_YAML: dict[str, Any] = {
    "models": list(VALID_MODEL_TYPES),
    "fusion_methods": list(VALID_FUSION_METHODS),
    "scales": list(VALID_SCALES),
    "stages": _STAGE_DEFS,
    "default_stages": ["autoencoder", "curriculum", "fusion"],
    "ckpt_stages": {
        "vgae": "autoencoder",
        "dgi": "autoencoder",
        "gat": "curriculum",
        "temporal": "temporal",
        "fusion": "fusion",
    },
}

STAGE_MODEL_MAP: dict[str, str] = {k: v[1] for k, v in STAGES.items()}
STAGE_DEPENDENCIES: dict[str, list[tuple[str, str]]] = {
    name: [(d["model"], d["stage"]) for d in s.get("depends_on", [])]
    for name, s in _STAGE_DEFS.items()
    if s.get("depends_on")
}

for _family in VALID_MODEL_TYPES:
    _base = CONFIG_DIR / "models" / _family / "base.yaml"
    if not _base.exists():
        raise FileNotFoundError(f"Missing model base config: {_base}")
    for _scale in VALID_SCALES:
        _scale_file = CONFIG_DIR / "models" / _family / "scales" / f"{_scale}.yaml"
        if not _scale_file.exists():
            raise FileNotFoundError(f"Missing model scale config: {_scale_file}")

_fusion_base = CONFIG_DIR / "fusion" / "base.yaml"
if not _fusion_base.exists():
    raise FileNotFoundError(f"Missing fusion base config: {_fusion_base}")
for _scale in VALID_SCALES:
    _scale_file = CONFIG_DIR / "fusion" / "scales" / f"{_scale}.yaml"
    if not _scale_file.exists():
        raise FileNotFoundError(f"Missing fusion scale config: {_scale_file}")
for _method in VALID_FUSION_METHODS:
    _method_file = CONFIG_DIR / "fusion" / "methods" / f"{_method}.yaml"
    if not _method_file.exists():
        raise FileNotFoundError(f"Missing fusion method config: {_method_file}")

for _family in [*VALID_MODEL_TYPES, "fusion"]:
    _profile = CONFIG_DIR / "resources" / "profiles" / f"{_family}.yaml"
    if not _profile.exists():
        raise FileNotFoundError(f"Missing resource profile: {_profile}")
