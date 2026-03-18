"""Tests for manifest, catalog, and locking modules.

Tests cover lake path functions, manifest read/write/verify, cache locking,
and catalog rebuild (with DuckDB).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# ============================================================================
# Lake path functions (formerly LakeConfig)
# ============================================================================


class TestLakePathFunctions:
    def test_lake_root_from_env_returns_none_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KD_GAT_LAKE_ROOT", None)
            from graphids.config import lake_root_from_env

            assert lake_root_from_env() is None

    def test_lake_root_from_env_returns_path_when_set(self, tmp_path):
        with patch.dict(os.environ, {"KD_GAT_LAKE_ROOT": str(tmp_path)}):
            from graphids.config import lake_root_from_env

            root = lake_root_from_env()
            assert root is not None
            assert root == tmp_path

    def test_lake_run_dir_production(self, tmp_path):
        from graphids.config import lake_run_dir

        d = lake_run_dir(tmp_path, "hcrl_sa", "vgae", "large", "autoencoder", seed=42, production=True)
        assert d == tmp_path / "production" / "hcrl_sa" / "vgae_large_autoencoder" / "seed_42"

    def test_lake_run_dir_with_aux(self, tmp_path):
        from graphids.config import lake_run_dir

        d = lake_run_dir(
            tmp_path, "hcrl_sa", "gat", "small", "curriculum", aux="kd_standard", seed=123
        )
        assert (
            d
            == tmp_path / "production" / "hcrl_sa" / "gat_small_curriculum_kd_standard" / "seed_123"
        )

    def test_lake_run_dir_evaluation_uses_eval(self, tmp_path):
        from graphids.config import lake_run_dir

        d = lake_run_dir(tmp_path, "hcrl_sa", "vgae", "large", "evaluation", seed=42)
        assert "eval_large_evaluation" in str(d)

    def test_lake_run_dir_dev(self, tmp_path):
        from graphids.config import lake_run_dir

        d = lake_run_dir(tmp_path, "hcrl_sa", "vgae", "large", "autoencoder", production=False)
        assert "dev/" in str(d)

    def test_lake_cache_dir(self, tmp_path):
        from graphids.config import lake_cache_dir

        d = lake_cache_dir(tmp_path, "hcrl_sa", version="3.0.0")
        assert d == tmp_path / "cache" / "v3.0.0" / "hcrl_sa"

    def test_lake_catalog_path(self, tmp_path):
        from graphids.config import lake_catalog_path

        assert lake_catalog_path(tmp_path) == tmp_path / "catalog" / "kd_gat.duckdb"

    def test_lake_sweep_dir(self, tmp_path):
        from graphids.config import lake_sweep_dir

        assert lake_sweep_dir(tmp_path, "hcrl_sa") == tmp_path / "sweeps" / "hcrl_sa"

    def test_lake_exports_dir(self, tmp_path):
        from graphids.config import lake_exports_dir

        assert lake_exports_dir(tmp_path) == tmp_path / "exports"


# ============================================================================
# Manifest
# ============================================================================


class TestManifest:
    def _make_run_dir(self, tmp_path: Path) -> Path:
        """Create a minimal run directory with config.json and best_model.pt."""
        run_dir = tmp_path / "hcrl_sa" / "vgae_large_autoencoder" / "seed_42"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text('{"model_type": "vgae"}')
        (run_dir / "best_model.pt").write_bytes(b"fake model data")
        return run_dir

    def test_write_manifest(self, tmp_path):
        from graphids.pipeline.manifest import read_manifest, write_manifest

        run_dir = self._make_run_dir(tmp_path)
        path = write_manifest(
            run_dir,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
            stage="autoencoder",
            seed=42,
        )
        assert path.exists()
        assert path.name == "_manifest.json"

        manifest = read_manifest(run_dir)
        assert manifest is not None
        assert manifest.dataset == "hcrl_sa"
        assert manifest.model_type == "vgae"
        assert len(manifest.artifacts) == 2  # config, best_model

    def test_verify_manifest_ok(self, tmp_path):
        from graphids.pipeline.manifest import verify_manifest, write_manifest

        run_dir = self._make_run_dir(tmp_path)
        write_manifest(run_dir, "hcrl_sa", "vgae", "large", "autoencoder")
        ok, errors = verify_manifest(run_dir)
        assert ok
        assert errors == []

    def test_verify_manifest_missing_file(self, tmp_path):
        from graphids.pipeline.manifest import verify_manifest, write_manifest

        run_dir = self._make_run_dir(tmp_path)
        write_manifest(run_dir, "hcrl_sa", "vgae", "large", "autoencoder")

        # Delete an artifact
        (run_dir / "best_model.pt").unlink()

        ok, errors = verify_manifest(run_dir)
        assert not ok
        assert any("Missing" in e for e in errors)

    def test_verify_manifest_checksum_mismatch(self, tmp_path):
        from graphids.pipeline.manifest import verify_manifest, write_manifest

        run_dir = self._make_run_dir(tmp_path)
        write_manifest(run_dir, "hcrl_sa", "vgae", "large", "autoencoder")

        # Corrupt an artifact
        (run_dir / "best_model.pt").write_bytes(b"corrupted data")

        ok, errors = verify_manifest(run_dir)
        assert not ok
        assert any("Checksum mismatch" in e for e in errors)

    def test_read_manifest_missing(self, tmp_path):
        from graphids.pipeline.manifest import read_manifest

        assert read_manifest(tmp_path) is None


# ============================================================================
# Locking
# ============================================================================


class TestCacheLock:
    def test_lock_creates_lockfile(self, tmp_path):
        from graphids.core.preprocessing._locking import cache_lock

        cache_dir = tmp_path / "hcrl_sa"
        cache_dir.mkdir()

        with cache_lock(cache_dir):
            lock_file = tmp_path / ".hcrl_sa.lock"
            assert lock_file.exists()

    def test_lock_is_reentrant_different_dirs(self, tmp_path):
        from graphids.core.preprocessing._locking import cache_lock

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        with cache_lock(dir_a):
            with cache_lock(dir_b):
                pass  # Should not deadlock


# ============================================================================
# Catalog
# ============================================================================


class TestCatalog:
    def test_rebuild_catalog_empty(self, tmp_path):
        """Rebuild with no manifests should create catalog file path."""
        from graphids.pipeline.catalog import rebuild_catalog

        lake_root = tmp_path / "lake"
        (lake_root / "production").mkdir(parents=True)
        catalog_path = rebuild_catalog(lake_root)
        assert catalog_path == lake_root / "catalog" / "kd_gat.duckdb"

    def test_rebuild_and_query(self, tmp_path):
        """Full round-trip: write manifest → rebuild catalog → query."""
        from graphids.pipeline.catalog import catalog_status, rebuild_catalog
        from graphids.pipeline.manifest import write_manifest

        lake_root = tmp_path / "lake"
        run_dir = lake_root / "production" / "hcrl_sa" / "vgae_large_autoencoder" / "seed_42"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "vgae",
                    "training": {"lr": 0.003, "max_epochs": 300, "batch_size": 4096},
                }
            )
        )
        (run_dir / "metrics.json").write_text(json.dumps({"f1_macro": 0.95, "accuracy": 0.98}))
        (run_dir / "best_model.pt").write_bytes(b"model")

        write_manifest(run_dir, "hcrl_sa", "vgae", "large", "autoencoder", seed=42)

        catalog_path = rebuild_catalog(lake_root)

        status = catalog_status(catalog_path)
        assert status["exists"]
        assert status["total_runs"] == 1
        assert "autoencoder" in status["by_stage"]
