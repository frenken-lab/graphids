"""Pipeline topology and path resolution.

Typed ``Topology`` model for ``topology.json``, ``PathContext`` for run_dir
construction, identity hashing, and dataset catalog.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, FilePath, TypeAdapter, computed_field, model_validator

from .constants import (
    CATALOG_SUBPATH,
    CKPT_SUBPATH,
    CONFIG_DIR,
    DATASET_REGISTRY_PATH,
    LAST_CKPT_SUBPATH,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
    VALID_FUSION_METHODS,
    VALID_MODEL_FAMILIES,
    VALID_SCALES,
)

_CONFIGS_DIR = PROJECT_ROOT / "configs"
_file_check = TypeAdapter(FilePath)


class StageDef(BaseModel):
    model_config = ConfigDict(frozen=True)
    learning_type: str
    family: str
    mode: str
    depends_on: list[dict[str, str]] = []
    identity_keys: list[str]
    stage_tlas: list[str]


class Topology(BaseModel):
    model_config = ConfigDict(frozen=True)
    stages: dict[str, StageDef]
    default_stages: list[str]
    ckpt_stages: dict[str, str]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def stage_family_map(self) -> dict[str, str]:
        return {name: s.family for name, s in self.stages.items()}

    @model_validator(mode="after")
    def _validate_config_tree(self) -> Topology:
        required = [
            *(_CONFIGS_DIR / "models" / f"{f}.libsonnet" for f in VALID_MODEL_FAMILIES),
            _CONFIGS_DIR / "models" / "fusion" / "base.libsonnet",
            *(
                _CONFIGS_DIR / "models" / "fusion" / "methods" / f"{m}.libsonnet"
                for m in VALID_FUSION_METHODS
            ),
            *(_CONFIGS_DIR / "stages" / f"{s}.jsonnet" for s in self.stages),
        ]
        for p in required:
            _file_check.validate_python(p)
        _validate_submit_profiles(self)
        return self


def _validate_submit_profiles(topology: Topology) -> None:
    """Import-time shape check for ``configs/resources/submit_profiles.json``.

    Catches three drift modes at package load instead of sbatch time:

    1. ``stages`` entry in a composed profile not declared in ``stage_profiles``.
    2. ``stage_profiles`` entry missing required ``cpus`` / ``scaling`` fields.
    3. ``scale_mult`` key outside ``VALID_SCALES``.
    """
    path = _CONFIGS_DIR / "resources" / "submit_profiles.json"
    cfg = json.loads(path.read_text())
    stage_profiles = cfg.get("stage_profiles", {})
    for name, sp in stage_profiles.items():
        if "cpus" not in sp:
            raise ValueError(f"stage_profiles[{name!r}] missing 'cpus'")
        sc = sp.get("scaling")
        if not sc or "time_min" not in sc or "mem_gb" not in sc:
            raise ValueError(
                f"stage_profiles[{name!r}] must have scaling.time_min and scaling.mem_gb"
            )
        for block_name, block in (("time_min", sc["time_min"]), ("mem_gb", sc["mem_gb"])):
            bad = set((block.get("scale_mult") or {}).keys()) - VALID_SCALES
            if bad:
                raise ValueError(
                    f"stage_profiles[{name!r}].scaling.{block_name}.scale_mult has "
                    f"unknown scale(s) {sorted(bad)}; valid: {sorted(VALID_SCALES)}"
                )
    for name, p in cfg["submit_profiles"].items():
        if "stages" in p:
            unknown = [s for s in p["stages"] if s not in stage_profiles]
            if unknown:
                raise ValueError(
                    f"submit_profiles[{name!r}].stages refers to unknown stage_profiles "
                    f"{unknown}; declared: {sorted(stage_profiles)}"
                )


TOPOLOGY = Topology.model_validate_json((CONFIG_DIR / "matrix" / "topology.json").read_bytes())


def _walk(cfg: Any, dotted_key: str):
    cur = cfg
    for part in dotted_key.split("."):
        if cur is None:
            return None
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
    return cur


def compute_identity_hash(stage: str, cfg: Any) -> str:
    # Unknown stages raise immediately; an empty-string fallback used to collide
    # silently across typos and produce bogus run dirs.
    stage_def = TOPOLOGY.stages[stage]
    keys = stage_def.identity_keys
    unresolved = [k for k in keys if _walk(cfg, k) is None]
    if unresolved:
        raise KeyError(f"Identity keys {unresolved} missing for stage '{stage}'")
    pairs = [f"{k}={_walk(cfg, k)}" for k in sorted(keys)]
    return "_" + hashlib.sha256("|".join(pairs).encode()).hexdigest()[:8]


class PathContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    lake_root: str
    user: str
    dataset: str
    model_type: str
    scale: str
    stage: str
    identity: str
    kd_tag: str
    seed: int

    @property
    def run_dir(self) -> Path:
        return Path(
            f"{self.lake_root}/dev/{self.user}/{self.dataset}/"
            f"{self.model_type}_{self.scale}_{self.stage}"
            f"{self.identity}{self.kd_tag}/seed_{self.seed}"
        )

    @property
    def ckpt_file(self) -> Path:
        return self.run_dir / CKPT_SUBPATH

    @property
    def last_ckpt_file(self) -> Path:
        return self.run_dir / LAST_CKPT_SUBPATH

    def resolve_ckpt(self) -> Path:
        return self.ckpt_file if self.ckpt_file.exists() else self.last_ckpt_file


def data_dir(lake_root: str, dataset: str) -> Path:
    return Path(lake_root) / "raw" / dataset


def cache_dir(lake_root: str, dataset: str) -> Path:
    return Path(lake_root) / "cache" / f"v{PREPROCESSING_VERSION}" / dataset


def catalog_path(lake_root: str) -> Path:
    return Path(lake_root) / CATALOG_SUBPATH


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, dict[str, Any]]:
    if not DATASET_REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Dataset registry missing: {DATASET_REGISTRY_PATH}")
    registry = json.loads(DATASET_REGISTRY_PATH.read_text())
    return {
        name: {"name": name, "domain": domain, **entry}
        for domain, datasets in registry.items()
        if isinstance(datasets, dict)
        for name, entry in datasets.items()
    }


def dataset_names() -> list[str]:
    return [k for k in load_catalog() if not k.startswith("_")]
