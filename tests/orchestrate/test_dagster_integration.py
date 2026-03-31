"""Layer 2: Dagster integration tests — multi-asset graph materialization.

Tests the full asset graph: assets materialize in topological order,
IOManager sidecars are created, and asset checks validate checkpoint
state. Uses patched submit/poll that create checkpoint files on disk.

CLI arg correctness is NOT tested here — that's Layer 0 (build_cli_args).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import dagster as dg
import pytest

from graphids.config import run_dir
from graphids.orchestrate.component import (
    CheckpointPathIOManager,
    SlurmTrainingResource,
    _make_asset,
    _make_checkpoint_checks,
)

from .conftest import PARTITION_KEY

pytestmark = pytest.mark.dagster


# ---------------------------------------------------------------------------
# Fake SLURM backend
# ---------------------------------------------------------------------------


def _fake_submit(script, resources, *, job_name, dry_run=False):
    """Fake sbatch: parse --trainer.default_root_dir from script to create checkpoint.

    This token format is stable — it's how LightningCLI receives the run dir,
    tested independently in Layer 0 (test_build_cli_args_base_args).
    """
    for token in script.split():
        if token.startswith("--trainer.default_root_dir="):
            rd = Path(token.split("=", 1)[1])
            (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
            (rd / "checkpoints" / "best_model.ckpt").write_text("fake")
            break
    return 99999


@pytest.fixture()
def fake_slurm():
    """Context-managed patches for submit, poll, and sacct_query."""
    with mock.patch("graphids.orchestrate.component.submit", side_effect=_fake_submit), \
         mock.patch("graphids.orchestrate.component.poll", return_value="COMPLETED"), \
         mock.patch("graphids.orchestrate.component.sacct_query", return_value=""):
        yield


@pytest.fixture()
def pipeline_resources(lake_root):
    return {
        "slurm": SlurmTrainingResource(dry_run=False, poll_interval=0),
        "io_manager": CheckpointPathIOManager(base_dir=f"{lake_root}/.dagster/io"),
    }


def _build_assets(configs, partitions_def, lake_root):
    return [_make_asset(cfg, partitions_def, lake_root, "testuser") for cfg in configs]


# ---------------------------------------------------------------------------
# Graph materialization
# ---------------------------------------------------------------------------


def test_three_stage_pipeline_materializes(
    three_stage_configs, partitions_def, lake_root, fake_slurm, pipeline_resources,
):
    assets = _build_assets(three_stage_configs, partitions_def, lake_root)

    result = dg.materialize(
        assets, partition_key=PARTITION_KEY, resources=pipeline_resources,
    )

    assert result.success


def test_each_stage_writes_iodir_sidecar(
    three_stage_configs, partitions_def, lake_root, fake_slurm, pipeline_resources,
):
    assets = _build_assets(three_stage_configs, partitions_def, lake_root)
    dg.materialize(assets, partition_key=PARTITION_KEY, resources=pipeline_resources)

    sidecars = list((Path(lake_root) / ".dagster" / "io").rglob("*.json"))
    assert len(sidecars) == 3
    for sc in sidecars:
        data = json.loads(sc.read_text())
        assert "best_model.ckpt" in data["checkpoint_path"]


def test_each_stage_creates_checkpoint_and_marker(
    three_stage_configs, partitions_def, lake_root, fake_slurm, pipeline_resources,
):
    dg.materialize(
        _build_assets(three_stage_configs, partitions_def, lake_root),
        partition_key=PARTITION_KEY, resources=pipeline_resources,
    )

    for cfg in three_stage_configs:
        rd = run_dir(lake_root, "testuser", "set_01", cfg.model_type, cfg.scale,
                     cfg.stage, cfg.identity, cfg.kd_tag, 42)
        assert (Path(rd) / "checkpoints" / "best_model.ckpt").exists()
        assert (Path(rd) / ".complete").exists()


# ---------------------------------------------------------------------------
# Asset checks
# ---------------------------------------------------------------------------


def _stub_asset(name: str, partitions_def):
    """Stub asset providing partition context for checks."""
    @dg.asset(name=name, partitions_def=partitions_def)
    def _noop(context):
        return ""
    return _noop


def test_checkpoint_check_passes_when_complete(
    three_stage_configs, partitions_def, lake_root, completed_checkpoint,
):
    ae = three_stage_configs[0]
    completed_checkpoint("vgae", "small", "autoencoder", "_a1b2c3d4")

    checks = _make_checkpoint_checks({ae.asset_name: ae}, partitions_def, lake_root, "testuser")
    stub = _stub_asset(ae.asset_name, partitions_def)

    result = dg.materialize([stub] + checks, partition_key=PARTITION_KEY)

    assert result.success
    assert result.get_asset_check_evaluations()[0].passed


def test_checkpoint_check_fails_when_no_checkpoint(
    three_stage_configs, partitions_def, lake_root,
):
    ae = three_stage_configs[0]

    checks = _make_checkpoint_checks({ae.asset_name: ae}, partitions_def, lake_root, "testuser")
    stub = _stub_asset(ae.asset_name, partitions_def)

    result = dg.materialize(
        [stub] + checks, partition_key=PARTITION_KEY, raise_on_error=False,
    )

    assert not result.get_asset_check_evaluations()[0].passed


def test_checkpoint_check_fails_without_complete_marker(
    three_stage_configs, partitions_def, lake_root,
):
    """Checkpoint file exists but no .complete marker -> stale, check fails."""
    ae = three_stage_configs[0]
    rd = run_dir(lake_root, "testuser", "set_01", "vgae", "small",
                 "autoencoder", "_a1b2c3d4", "", 42)
    (Path(rd) / "checkpoints").mkdir(parents=True)
    (Path(rd) / "checkpoints" / "best_model.ckpt").touch()

    checks = _make_checkpoint_checks({ae.asset_name: ae}, partitions_def, lake_root, "testuser")
    stub = _stub_asset(ae.asset_name, partitions_def)

    result = dg.materialize(
        [stub] + checks, partition_key=PARTITION_KEY, raise_on_error=False,
    )

    assert not result.get_asset_check_evaluations()[0].passed
