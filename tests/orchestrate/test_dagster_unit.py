"""Layer 1: Dagster unit tests — single-asset materialization behavior.

Tests dagster-specific behaviors: skip-when-complete, dry-run, failure
handling, and IOManager checkpoint handoff. CLI arg construction is
tested in Layer 0 (test_pure.py::test_build_cli_args_*).
"""

from __future__ import annotations

from unittest import mock

import dagster as dg
import pytest

from graphids.orchestrate.component import (
    CheckpointPathIOManager,
    SlurmTrainingResource,
    _make_asset,
    build_cli_args,
)

from .conftest import PARTITION_KEY

pytestmark = pytest.mark.dagster


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def test_dry_run_materializes_without_subprocess(
    autoencoder_cfg, partitions_def, dagster_resources,
):
    asset_fn = _make_asset(autoencoder_cfg, partitions_def, lake_root_from(dagster_resources), "testuser")

    result = dg.materialize(
        [asset_fn], partition_key=PARTITION_KEY, resources=dagster_resources,
    )

    assert result.success


# ---------------------------------------------------------------------------
# Skip-when-complete
# ---------------------------------------------------------------------------


def test_skip_when_checkpoint_and_marker_exist(
    autoencoder_cfg, partitions_def, dagster_resources, completed_checkpoint,
):
    """Asset returns existing checkpoint without calling generate_script."""
    completed_checkpoint("vgae", "small", "autoencoder", "_abc12345")
    asset_fn = _make_asset(autoencoder_cfg, partitions_def, lake_root_from(dagster_resources), "testuser")

    with mock.patch("graphids.orchestrate.component.generate_script") as gen_spy:
        result = dg.materialize(
            [asset_fn], partition_key=PARTITION_KEY, resources=dagster_resources,
        )

    assert result.success
    gen_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


def test_slurm_failure_causes_materialization_failure(
    autoencoder_cfg, partitions_def, lake_root,
):
    asset_fn = _make_asset(autoencoder_cfg, partitions_def, lake_root, "testuser")

    with mock.patch("graphids.orchestrate.component.submit", return_value=99999), \
         mock.patch("graphids.orchestrate.component.poll", return_value="FAILED"), \
         mock.patch("graphids.orchestrate.component.sacct_query", return_value=""):
        result = dg.materialize(
            [asset_fn],
            partition_key=PARTITION_KEY,
            resources={
                "slurm": SlurmTrainingResource(dry_run=False, poll_interval=0),
                "io_manager": CheckpointPathIOManager(base_dir=f"{lake_root}/.dagster/io"),
            },
            raise_on_error=False,
        )

    assert not result.success


# ---------------------------------------------------------------------------
# IOManager checkpoint handoff
# ---------------------------------------------------------------------------


def test_iomanager_passes_upstream_ckpt_to_build_cli_args(
    autoencoder_cfg, curriculum_cfg, partitions_def,
    dagster_resources, completed_checkpoint,
):
    """Downstream asset receives upstream checkpoint via IOManager -> build_cli_args."""
    completed_checkpoint("vgae", "small", "autoencoder", "_abc12345")

    upstream_fn = _make_asset(autoencoder_cfg, partitions_def, lake_root_from(dagster_resources), "testuser")
    downstream_fn = _make_asset(curriculum_cfg, partitions_def, lake_root_from(dagster_resources), "testuser")

    with mock.patch(
        "graphids.orchestrate.component.build_cli_args", wraps=build_cli_args,
    ) as cli_spy:
        result = dg.materialize(
            [upstream_fn, downstream_fn],
            partition_key=PARTITION_KEY,
            resources=dagster_resources,
        )

    assert result.success
    # Upstream was skipped (already complete), so only downstream called build_cli_args
    assert cli_spy.call_count == 1
    # Verify upstream_ckpts kwarg contains the autoencoder checkpoint
    _, kwargs = cli_spy.call_args
    upstream_ckpts = kwargs.get("upstream_ckpts", cli_spy.call_args[0][4])
    assert "autoencoder_abc12345" in upstream_ckpts
    assert "best_model.ckpt" in upstream_ckpts["autoencoder_abc12345"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def lake_root_from(resources: dict) -> str:
    """Extract lake_root from dagster_resources fixture."""
    io_mgr = resources["io_manager"]
    # base_dir is "{lake_root}/.dagster/io"
    return str(io_mgr.base_dir).rsplit("/.dagster/io", 1)[0]
