"""Path composition, identity hashing, and lightweight resolvers.

Pure path helpers (``compute_identity_hash``, ``checkpoint_path``,
``data_dir``, ``cache_dir``) plus thin resolvers that read but never
write: dataset catalog, checkpoint probes, run-dir identity parsing,
and the lake-write env-var gate.

Filesystem *writes* (run records, config snapshots, atomic-write
primitive) live in ``graphids.core.io``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .constants import (
    CATALOG_SUBPATH,
    DATASET_REGISTRY_PATH,
    PREPROCESSING_VERSION,
)
from .schemas import PathContext
from .topology import PIPELINE_TOPOLOGY, STAGES


def compute_preprocessing_hash() -> str:
    import hashlib

    from graphids.core.data.datasets.can_bus import (
        N_EDGE_FEATURES,
        N_NODE_FEATURES,
    )

    components = [
        PREPROCESSING_VERSION,
        str(N_NODE_FEATURES),
        str(N_EDGE_FEATURES),
        "100",
        "100",
        "0.8",
    ]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]


def data_dir(lake_root: str, dataset: str) -> Path:
    candidate = Path(lake_root) / "raw" / dataset
    if candidate.exists():
        return candidate
    return Path("data") / "automotive" / dataset


def cache_dir(lake_root: str, dataset: str) -> Path:
    return Path(lake_root) / "cache" / f"v{PREPROCESSING_VERSION}" / dataset


def compute_identity_hash(stage: str, cfg: Any) -> str:
    import hashlib

    stage_def = PIPELINE_TOPOLOGY.get("stages", {}).get(stage, {})
    keys = stage_def.get("identity_keys", [])
    if not keys:
        return ""

    def _get(dotted_key: str, default=None):
        cur = cfg
        for part in dotted_key.split("."):
            if cur is None:
                return default
            cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        return cur if cur is not None else default

    unresolved = [k for k in keys if _get(k) is None]
    if unresolved:
        raise KeyError(
            f"Identity keys {unresolved} not found in config for stage '{stage}'. "
            "These keys must be set for stable checkpoint identity."
        )

    pairs = [f"{k}={_get(k, '_default_')}" for k in sorted(keys)]
    return "_" + hashlib.sha256("|".join(pairs).encode()).hexdigest()[:8]


def checkpoint_path(
    lake_root: str,
    dataset: str,
    model_type: str,
    scale: str,
    seed: int,
    cfg: Any,
    *,
    gat_stage: str = "supervised",
) -> Path:
    """Resolve checkpoint path for a given model. Delegates to PathContext."""
    stage = PIPELINE_TOPOLOGY.get("ckpt_stages", {}).get(model_type, model_type)
    if model_type == "gat":
        stage = gat_stage
    identity = compute_identity_hash(stage, cfg)
    return PathContext(
        lake_root=lake_root,
        user=os.environ.get("USER", "unknown"),
        dataset=dataset,
        model_type=model_type,
        scale=scale,
        stage=stage,
        identity=identity,
        kd_tag="",
        seed=seed,
    ).ckpt_file


# -----------------------------------------------------------------------------
# Lake-write gate
# -----------------------------------------------------------------------------


class LakeWriteError(PermissionError):
    """Raised when lake write is attempted without KD_GAT_LAKE_WRITE=1."""


def require_lake_write() -> None:
    """Gate all data-lake writes behind KD_GAT_LAKE_WRITE=1.

    SLURM jobs get this via _preamble.sh → .env. Direct CLI invocations
    on login nodes or interactive sessions are blocked by default.
    """
    if os.environ.get("KD_GAT_LAKE_WRITE") != "1":
        raise LakeWriteError(
            "Lake write blocked: set KD_GAT_LAKE_WRITE=1 in environment "
            "(SLURM jobs get this from .env via _preamble.sh). "
            "For read-only runs, use --dry-run."
        )


# -----------------------------------------------------------------------------
# Dataset catalog
# -----------------------------------------------------------------------------

_catalog_cache: dict[str, dict[str, Any]] | None = None


def catalog_path(lake_root: str) -> Path:
    """Return the DuckDB experiment catalog path for a given lake root."""
    return Path(lake_root) / CATALOG_SUBPATH


def load_catalog() -> dict[str, dict[str, Any]]:
    """Load dataset catalog from ``configs/datasets/dataset_registry.json``.

    The on-disk registry is domain-nested (``{"automotive": {"hcrl_ch":
    {...}}}``). This function flattens to ``{dataset_name: {metadata_dict}}``
    and injects ``entry["name"]`` from the dict key so downstream consumers
    don't need to care about the domain layer.
    """
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache
    if not DATASET_REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Dataset registry missing: {DATASET_REGISTRY_PATH}")
    registry = json.loads(DATASET_REGISTRY_PATH.read_text())
    catalog: dict[str, dict[str, Any]] = {}
    for domain, datasets in registry.items():
        if not isinstance(datasets, dict):
            continue
        for name, entry in datasets.items():
            flat = {"name": name, "domain": domain, **entry}
            catalog[name] = flat
    _catalog_cache = catalog
    return catalog


def dataset_names() -> list[str]:
    """Return list of dataset names (for dagster partitions, etc.)."""
    return [k for k in load_catalog() if not k.startswith("_")]


# -----------------------------------------------------------------------------
# Checkpoint resolution
# -----------------------------------------------------------------------------


def resolve_checkpoint(path_ctx: PathContext) -> Path:
    """Best-available checkpoint: ``ckpt_file`` if present, else ``last_ckpt_file``.

    The only filesystem probe outside of a write path — lives here rather
    than on ``PathContext`` so the Pydantic model stays side-effect free.
    """
    return path_ctx.ckpt_file if path_ctx.ckpt_file.exists() else path_ctx.last_ckpt_file


# -----------------------------------------------------------------------------
# Run-dir identity parser
# -----------------------------------------------------------------------------


def parse_identity_from_run_dir(run_dir: str) -> dict[str, Any]:
    """Extract identity fields from a run_dir path.

    Path convention:
        {lake_root}/dev/{user}/{dataset}/{model}_{scale}_{stage}{identity}{kd_tag}/seed_{seed}
    """
    parts = Path(run_dir).parts
    seed_part = parts[-1]  # "seed_42"
    seed = int(seed_part.split("_", 1)[1])
    dir_name = parts[-2]
    dataset = parts[-3]
    user = parts[-4]

    kd_tag = ""
    if dir_name.endswith("_kd"):
        kd_tag = "_kd"
        dir_name = dir_name[: -len("_kd")]

    last_underscore = dir_name.rfind("_")
    identity_hash = "_" + dir_name[last_underscore + 1 :]
    remainder = dir_name[:last_underscore]

    stage = ""
    for s in STAGES:
        suffix = f"_{s}"
        if remainder.endswith(suffix):
            stage = s
            remainder = remainder[: -len(suffix)]
            break

    last_us = remainder.rfind("_")
    model_type = remainder[:last_us]
    scale = remainder[last_us + 1 :]

    return {
        "dataset": dataset,
        "user": user,
        "seed": seed,
        "model_family": model_type,
        "scale": scale,
        "stage": stage,
        "identity_hash": identity_hash,
        "kd_tag": kd_tag,
    }
