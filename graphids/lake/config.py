"""Lake configuration: env var resolution + path derivation.

LakeConfig is a frozen Pydantic model that resolves $KD_GAT_LAKE_ROOT
and derives all lake paths from identity dimensions.

Layer placement: imports from graphids.config only (Layer 1).
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

from pydantic import BaseModel, Field


class LakeConfig(BaseModel, frozen=True):
    """Data lake configuration, resolved from environment."""

    lake_root: Path
    path_version: int = Field(1, ge=1)

    @classmethod
    def from_env(cls) -> LakeConfig | None:
        """Resolve from $KD_GAT_LAKE_ROOT. Returns None if unset."""
        root = os.environ.get("KD_GAT_LAKE_ROOT")
        if not root:
            return None
        return cls(lake_root=Path(root))

    # ------------------------------------------------------------------
    # Path derivation
    # ------------------------------------------------------------------

    def run_dir(
        self,
        dataset: str,
        model_type: str,
        scale: str,
        stage: str,
        aux: str = "",
        seed: int = 42,
        production: bool = True,
    ) -> Path:
        """Derive the full run directory from identity dimensions.

        Path: {lake_root}/{production|dev/user}/{dataset}/{model}_{scale}_{stage}[_{aux}]/seed_{seed}
        """
        tier = "production" if production else f"dev/{getpass.getuser()}"
        model = "eval" if stage == "evaluation" else model_type
        suffix = f"_{aux}" if aux else ""
        return self.lake_root / tier / dataset / f"{model}_{scale}_{stage}{suffix}" / f"seed_{seed}"

    def cache_dir(self, dataset: str, version: str | None = None) -> Path:
        """Cache directory for preprocessed graphs."""
        if version is None:
            from graphids.config import PREPROCESSING_VERSION

            version = PREPROCESSING_VERSION
        return self.lake_root / "cache" / f"v{version}" / dataset

    def raw_dir(self, dataset: str) -> Path:
        """Raw dataset directory."""
        return self.lake_root / "raw" / dataset

    def sweep_dir(self, dataset: str) -> Path:
        """Sweep results directory for a dataset."""
        return self.lake_root / "sweeps" / dataset

    def catalog_path(self) -> Path:
        """DuckDB catalog file."""
        return self.lake_root / "catalog" / "kd_gat.duckdb"

    def exports_dir(self) -> Path:
        """Exports directory for parquet files."""
        return self.lake_root / "exports"

    def mlflow_uri(self) -> str:
        """MLflow tracking URI on ESS."""
        return f"sqlite:///{self.lake_root / 'mlflow' / 'mlflow.db'}"
