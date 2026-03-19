"""Path derivation for the ESS data lake.

Single source of truth for lake directory layout. No domain imports —
only stdlib + constants passed by callers.

Absorbed from graphids/config/paths.py (lake_run_dir, lake_cache_dir, etc.).
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

# Preprocessing version — duplicated here to avoid importing config/.
# The canonical value lives in config/constants.py; storage/paths.py uses
# this default which callers can override via the version= parameter.
_DEFAULT_PREPROCESSING_VERSION = "4"


def lake_run_dir(
    lake_root: str | Path,
    dataset: str,
    model_type: str,
    scale: str,
    stage: str,
    aux: str = "",
    seed: int = 42,
    production: bool = True,
) -> Path:
    """Derive a lake run directory from raw identity dimensions.

    Path: {lake_root}/{production|dev/user}/{dataset}/{model}_{scale}_{stage}[_{aux}]/seed_{seed}

    This is the single source of truth for lake run-dir layout.
    """
    tier = "production" if production else f"dev/{getpass.getuser()}"
    model = "eval" if stage == "evaluation" else model_type
    suffix = f"_{aux}" if aux else ""
    return Path(lake_root) / tier / dataset / f"{model}_{scale}_{stage}{suffix}" / f"seed_{seed}"


def lake_cache_dir(
    lake_root: str | Path,
    dataset: str,
    version: str | None = None,
) -> Path:
    """Derive a lake cache directory for preprocessed graphs.

    Path: {lake_root}/cache/v{version}/{dataset}
    """
    if version is None:
        version = _DEFAULT_PREPROCESSING_VERSION
    return Path(lake_root) / "cache" / f"v{version}" / dataset


def lake_raw_dir(lake_root: str | Path, dataset: str) -> Path:
    """Derive a lake raw-data directory.

    Path: {lake_root}/raw/{dataset}
    """
    return Path(lake_root) / "raw" / dataset


def lake_root_from_env() -> Path | None:
    """Read KD_GAT_LAKE_ROOT from the environment.

    Returns Path if set, None otherwise.
    """
    root = os.environ.get("KD_GAT_LAKE_ROOT")
    if not root:
        return None
    return Path(root)


def lake_sweep_dir(lake_root: str | Path, dataset: str) -> Path:
    """Sweep results directory for a dataset.

    Path: {lake_root}/sweeps/{dataset}
    """
    return Path(lake_root) / "sweeps" / dataset


def lake_catalog_path(lake_root: str | Path) -> Path:
    """DuckDB catalog file path.

    Path: {lake_root}/catalog/kd_gat.duckdb
    """
    return Path(lake_root) / "catalog" / "kd_gat.duckdb"


def lake_exports_dir(lake_root: str | Path) -> Path:
    """Exports directory for parquet files.

    Path: {lake_root}/exports
    """
    return Path(lake_root) / "exports"
