"""Pipeline topology and path resolution.

Typed ``Topology`` model for ``topology.json``, path models
(``PathContext``, ``RunDirIdentity``), identity hashing, and dataset catalog.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, FilePath, TypeAdapter, computed_field, model_validator

from .constants import (
    CATALOG_SUBPATH,
    CKPT_SUBPATH,
    COMPLETE_MARKER,
    CONFIG_DIR,
    DATASET_REGISTRY_PATH,
    LAST_CKPT_SUBPATH,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
    VALID_FUSION_METHODS,
    VALID_MODEL_FAMILIES,
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
            _CONFIGS_DIR / "resources" / "job_profiles.json",
        ]
        for p in required:
            _file_check.validate_python(p)
        profiles = json.loads((_CONFIGS_DIR / "resources" / "job_profiles.json").read_text())
        if missing := VALID_MODEL_FAMILIES - profiles.keys():
            raise ValueError(f"Missing resource profiles: {sorted(missing)}")
        return self


TOPOLOGY = Topology.model_validate_json((CONFIG_DIR / "matrix" / "topology.json").read_bytes())


def _walk(cfg: Any, dotted_key: str):
    cur = cfg
    for part in dotted_key.split("."):
        if cur is None:
            return None
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
    return cur


def compute_identity_hash(stage: str, cfg: Any) -> str:
    stage_def = TOPOLOGY.stages.get(stage)
    if stage_def is None:
        return ""
    keys = stage_def.identity_keys
    unresolved = [k for k in keys if _walk(cfg, k) is None]
    if unresolved:
        raise KeyError(f"Identity keys {unresolved} missing for stage '{stage}'")
    pairs = [f"{k}={_walk(cfg, k)}" for k in sorted(keys)]
    return "_" + hashlib.sha256("|".join(pairs).encode()).hexdigest()[:8]


class RunDirIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset: str
    user: str
    seed: int
    model_family: str
    scale: str
    stage: str
    identity_hash: str
    kd_tag: str = ""

    @classmethod
    def from_run_dir(cls, run_dir: str) -> RunDirIdentity:
        parts = Path(run_dir).parts
        seed = int(parts[-1].split("_", 1)[1])
        dir_name, dataset, user = parts[-2], parts[-3], parts[-4]
        kd_tag = ""
        if dir_name.endswith("_kd"):
            kd_tag, dir_name = "_kd", dir_name[:-3]
        last_us = dir_name.rfind("_")
        identity_hash, remainder = "_" + dir_name[last_us + 1 :], dir_name[:last_us]
        stage = ""
        for s in TOPOLOGY.stages:
            if remainder.endswith(f"_{s}"):
                stage, remainder = s, remainder[: -len(s) - 1]
                break
        split = remainder.rfind("_")
        return cls(
            dataset=dataset,
            user=user,
            seed=seed,
            model_family=remainder[:split],
            scale=remainder[split + 1 :],
            stage=stage,
            identity_hash=identity_hash,
            kd_tag=kd_tag,
        )


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
    def complete_marker(self) -> Path:
        return self.run_dir / COMPLETE_MARKER

    @property
    def last_ckpt_file(self) -> Path:
        return self.run_dir / LAST_CKPT_SUBPATH

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def resolve_ckpt(self) -> Path:
        return self.ckpt_file if self.ckpt_file.exists() else self.last_ckpt_file

    @classmethod
    def for_checkpoint(
        cls,
        lake_root: str,
        dataset: str,
        model_type: str,
        scale: str,
        seed: int,
        cfg: Any,
        *,
        gat_stage: str = "supervised",
    ) -> Path:
        stage = (
            gat_stage if model_type == "gat" else TOPOLOGY.ckpt_stages.get(model_type, model_type)
        )
        return cls(
            lake_root=lake_root,
            user=os.environ.get("USER", "unknown"),
            dataset=dataset,
            model_type=model_type,
            scale=scale,
            stage=stage,
            identity=compute_identity_hash(stage, cfg),
            kd_tag="",
            seed=seed,
        ).ckpt_file


def data_dir(lake_root: str, dataset: str) -> Path:
    candidate = Path(lake_root) / "raw" / dataset
    return candidate if candidate.exists() else Path("data") / "automotive" / dataset


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
