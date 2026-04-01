"""Shared fixtures for orchestrate tests.

Orchestrate modules have NO torch/Lightning imports, so all fixtures
are safe on the login node.
"""

from __future__ import annotations

from pathlib import Path

import dagster as dg
import pytest

from graphids.config import run_dir
from graphids.orchestrate.component import (
    CheckpointPathIOManager,
    SlurmTrainingResource,
    StageConfig,
)

PARTITION_KEY = dg.MultiPartitionKey({"dataset": "set_01", "seed": "42"})


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def lake_root(tmp_path):
    return str(tmp_path / "lake")


@pytest.fixture(scope="module")
def partitions_def():
    """Partition definition shared across tests. Stateless — module scope is safe."""
    return dg.MultiPartitionsDefinition({
        "dataset": dg.StaticPartitionsDefinition(["set_01"]),
        "seed": dg.StaticPartitionsDefinition(["42"]),
    })


@pytest.fixture()
def dagster_resources(lake_root):
    """Dry-run SLURM resource + IOManager backed by tmp_path."""
    return {
        "slurm": SlurmTrainingResource(dry_run=True, poll_interval=0),
        "io_manager": CheckpointPathIOManager(base_dir=f"{lake_root}/.dagster/io"),
    }


# ---------------------------------------------------------------------------
# StageConfig fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def autoencoder_cfg():
    """Root asset — no upstream deps."""
    return StageConfig(
        asset_name="autoencoder_abc12345",
        stage="autoencoder", model_type="vgae", scale="small",
        config_files=(
            "graphids/config/stages/autoencoder.yaml",
            "graphids/config/models/vgae/small.yaml",
        ),
        identity="_abc12345",
    )


@pytest.fixture()
def curriculum_cfg():
    """Downstream asset depending on autoencoder."""
    return StageConfig(
        asset_name="curriculum_def67890",
        stage="curriculum", model_type="gat", scale="small",
        config_files=("graphids/config/stages/curriculum.yaml",),
        model_overrides={"conv_type": "gatv2"},
        identity="_def67890",
        upstream_asset_names=("autoencoder_abc12345",),
        upstream_ckpt_flags={
            "autoencoder_abc12345": "--data.init_args.vgae_ckpt_path",
        },
    )


@pytest.fixture()
def three_stage_configs():
    """Autoencoder -> curriculum -> fusion pipeline."""
    ae = StageConfig(
        asset_name="autoencoder_a1b2c3d4",
        stage="autoencoder", model_type="vgae", scale="small",
        config_files=("graphids/config/stages/autoencoder.yaml",),
        identity="_a1b2c3d4",
    )
    cur = StageConfig(
        asset_name="curriculum_e5f6g7h8",
        stage="curriculum", model_type="gat", scale="small",
        config_files=("graphids/config/stages/curriculum.yaml",),
        model_overrides={"conv_type": "gatv2"},
        identity="_e5f6g7h8",
        upstream_asset_names=("autoencoder_a1b2c3d4",),
        upstream_ckpt_flags={
            "autoencoder_a1b2c3d4": "--data.init_args.vgae_ckpt_path",
        },
    )
    fus = StageConfig(
        asset_name="fusion_i9j0k1l2",
        stage="fusion", model_type="dqn", scale="small",
        config_files=("graphids/config/stages/fusion.yaml",),
        identity="_i9j0k1l2",
        upstream_asset_names=("autoencoder_a1b2c3d4", "curriculum_e5f6g7h8"),
        upstream_ckpt_flags={
            "autoencoder_a1b2c3d4": "--data.init_args.vgae_ckpt_path",
            "curriculum_e5f6g7h8": "--data.init_args.gat_ckpt_path",
        },
    )
    return ae, cur, fus


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def completed_checkpoint(lake_root):
    """Factory fixture: creates a checkpoint + .complete marker on disk.

    Usage::

        def test_something(completed_checkpoint):
            ckpt_path = completed_checkpoint("vgae", "small", "autoencoder", "_abc12345")
    """
    def _create(model_type, scale, stage, identity, kd_tag="", seed=42):
        rd = run_dir(lake_root, "testuser", "set_01", model_type, scale,
                     stage, identity, kd_tag, seed)
        rd_path = Path(rd)
        ckpt = rd_path / "checkpoints" / "best_model.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.touch()
        (rd_path / ".complete").touch()
        return str(ckpt)
    return _create
