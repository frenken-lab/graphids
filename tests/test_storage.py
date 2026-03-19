"""Tests for graphids.storage — gateway + mapper + paths."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from graphids.storage import StorageGateway, ArtifactMapper, open_gateway
from graphids.storage.paths import (
    lake_cache_dir,
    lake_catalog_path,
    lake_exports_dir,
    lake_raw_dir,
    lake_root_from_env,
    lake_run_dir,
    lake_sweep_dir,
)


# ---------------------------------------------------------------------------
# Path primitives
# ---------------------------------------------------------------------------


class TestLakePaths:
    def test_lake_run_dir_production(self):
        p = lake_run_dir("/lake", "hcrl_sa", "vgae", "large", "autoencoder", seed=42)
        assert p == Path("/lake/production/hcrl_sa/vgae_large_autoencoder/seed_42")

    def test_lake_run_dir_dev(self):
        p = lake_run_dir("/lake", "hcrl_sa", "vgae", "large", "autoencoder", production=False)
        # dev tier includes username
        assert "dev/" in str(p)
        assert "hcrl_sa/vgae_large_autoencoder/seed_42" in str(p)

    def test_lake_run_dir_with_aux(self):
        p = lake_run_dir("/lake", "hcrl_sa", "gat", "small", "curriculum", aux="kd_standard")
        assert "gat_small_curriculum_kd_standard" in str(p)

    def test_lake_run_dir_evaluation_uses_eval(self):
        p = lake_run_dir("/lake", "hcrl_sa", "vgae", "large", "evaluation")
        assert "eval_large_evaluation" in str(p)

    def test_lake_cache_dir(self):
        p = lake_cache_dir("/lake", "hcrl_sa")
        assert "cache/v" in str(p)
        assert str(p).endswith("/hcrl_sa")

    def test_lake_cache_dir_custom_version(self):
        p = lake_cache_dir("/lake", "hcrl_sa", version="5")
        assert "/cache/v5/" in str(p)

    def test_lake_raw_dir(self):
        assert lake_raw_dir("/lake", "hcrl_sa") == Path("/lake/raw/hcrl_sa")

    def test_lake_sweep_dir(self):
        assert lake_sweep_dir("/lake", "hcrl_sa") == Path("/lake/sweeps/hcrl_sa")

    def test_lake_catalog_path(self):
        assert lake_catalog_path("/lake") == Path("/lake/catalog/kd_gat.duckdb")

    def test_lake_exports_dir(self):
        assert lake_exports_dir("/lake") == Path("/lake/exports")


# ---------------------------------------------------------------------------
# StorageGateway
# ---------------------------------------------------------------------------


class TestStorageGateway:
    def test_init_raw_coords(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        p = gw.resolve("autoencoder")
        assert "vgae_large_autoencoder" in str(p)

    def test_init_requires_args(self):
        with pytest.raises(ValueError, match="requires either"):
            StorageGateway()

    def test_init_partial_raw_coords_fails(self):
        with pytest.raises(ValueError, match="requires either"):
            StorageGateway(lake_root="/tmp", dataset="hcrl_sa")

    def test_resolve_with_name(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        p = gw.resolve("autoencoder", "best_model.pt")
        assert p.name == "best_model.pt"

    def test_resolve_cross_model(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="gat",
            scale="large",
        )
        p = gw.resolve("autoencoder", "best_model.pt", model_type="vgae")
        assert "vgae_large_autoencoder" in str(p)

    def test_exists_false(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        assert not gw.exists("autoencoder", "best_model.pt")

    def test_exists_true(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        p = gw.resolve("autoencoder", "best_model.pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("test")
        assert gw.exists("autoencoder", "best_model.pt")

    def test_require_raises(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        with pytest.raises(FileNotFoundError):
            gw.require("autoencoder", "best_model.pt")

    def test_require_returns_path(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        p = gw.resolve("autoencoder", "best_model.pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("test")
        assert gw.require("autoencoder", "best_model.pt") == p

    def test_ensure_dir(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        d = gw.ensure_dir("autoencoder")
        assert d.exists()
        assert d.is_dir()

    def test_write_read_bytes(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        path = tmp_path / "test.bin"
        gw.write_bytes(path, b"hello")
        assert gw.read_bytes(path) == b"hello"

    def test_write_read_json(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        path = tmp_path / "test.json"
        data = {"key": "value", "nested": {"a": 1}}
        gw.write_json(path, data)
        assert gw.read_json(path) == data

    def test_list_artifacts(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        d = gw.ensure_dir("autoencoder")
        (d / "best_model.pt").write_text("x")
        (d / "config.json").write_text("{}")
        artifacts = gw.list_artifacts("autoencoder")
        assert sorted(artifacts) == ["best_model.pt", "config.json"]

    def test_list_artifacts_empty(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        assert gw.list_artifacts("autoencoder") == []

    def test_lock(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        d = gw.ensure_dir("autoencoder")
        with gw.lock(d):
            # Lock acquired — write something inside
            (d / "test.txt").write_text("locked write")
        assert (d / "test.txt").read_text() == "locked write"

    def test_atomic_write_survives_parent_missing(self, tmp_path):
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        path = tmp_path / "deep" / "nested" / "file.bin"
        gw.write_bytes(path, b"data")
        assert path.read_bytes() == b"data"


# ---------------------------------------------------------------------------
# ArtifactMapper
# ---------------------------------------------------------------------------


class TestArtifactMapper:
    def _make(self, tmp_path) -> tuple[StorageGateway, ArtifactMapper]:
        gw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
        )
        return gw, ArtifactMapper(gw)

    def test_save_load_checkpoint(self, tmp_path):
        gw, mapper = self._make(tmp_path)
        sd = {"weight": torch.randn(3, 4)}
        path = mapper.save_checkpoint(sd, "autoencoder")
        assert path.exists()
        loaded = mapper.load_checkpoint("autoencoder")
        assert torch.allclose(sd["weight"], loaded["weight"])

    def test_load_checkpoint_missing_raises(self, tmp_path):
        _, mapper = self._make(tmp_path)
        with pytest.raises(FileNotFoundError):
            mapper.load_checkpoint("autoencoder")

    def test_save_load_config(self, tmp_path):
        from graphids.config import PipelineConfig

        gw, mapper = self._make(tmp_path)
        cfg = PipelineConfig(dataset="hcrl_sa", model_type="vgae", scale="large")
        path = mapper.save_config(cfg, "autoencoder")
        assert path.exists()
        loaded = mapper.load_config("autoencoder")
        assert loaded.dataset == "hcrl_sa"
        assert loaded.model_type == "vgae"

    def test_save_dqn_checkpoint(self, tmp_path):
        gw, mapper = self._make(tmp_path)
        state = {
            "q_network": {"w": torch.randn(2, 2)},
            "target_network": {"w": torch.randn(2, 2)},
            "epsilon": 0.05,
        }
        path = mapper.save_dqn_checkpoint(state, "fusion")
        assert path.exists()

    def test_save_pickle(self, tmp_path):
        _, mapper = self._make(tmp_path)
        path = tmp_path / "vocab.pkl"
        data = {"a": 1, "b": 2}
        mapper.save_pickle(data, path)
        loaded = mapper.load_pickle(path)
        assert loaded == data

    def test_save_json(self, tmp_path):
        gw, mapper = self._make(tmp_path)
        data = {"key": "value"}
        path = mapper.save_json(data, "autoencoder", "metrics.json")
        assert path.exists()
        assert json.loads(path.read_text()) == data

    def test_save_npz(self, tmp_path):
        gw, mapper = self._make(tmp_path)
        data = {"arr": np.array([1, 2, 3])}
        path = mapper.save_npz(data, "autoencoder", "embeddings.npz")
        assert path.exists()
        loaded = np.load(path)
        assert np.array_equal(loaded["arr"], data["arr"])


# ---------------------------------------------------------------------------
# open_gateway convenience
# ---------------------------------------------------------------------------


class TestOpenGateway:
    def test_open_gateway(self, tmp_path):
        from graphids.config import PipelineConfig

        cfg = PipelineConfig(
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
            lake_root=str(tmp_path),
        )
        gw, mapper = open_gateway(cfg)
        assert isinstance(gw, StorageGateway)
        assert isinstance(mapper, ArtifactMapper)

    def test_cfg_init_matches_raw_init(self, tmp_path):
        from graphids.config import PipelineConfig

        cfg = PipelineConfig(
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
            lake_root=str(tmp_path),
            seed=42,
        )
        gw_cfg = StorageGateway(cfg=cfg)
        gw_raw = StorageGateway(
            lake_root=tmp_path,
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
            seed=42,
        )
        assert gw_cfg.resolve("autoencoder") == gw_raw.resolve("autoencoder")
