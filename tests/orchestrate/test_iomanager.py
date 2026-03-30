"""Layer 3: IOManager unit tests — checkpoint path sidecar read/write.

Tests CheckpointPathIOManager using dagster's build_output_context /
build_input_context helpers. No SLURM, no torch.
"""

from __future__ import annotations

import json

import dagster as dg
import pytest

from graphids.orchestrate.component import CheckpointPathIOManager


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def io_manager(tmp_path):
    return CheckpointPathIOManager(base_dir=str(tmp_path))


def _output_context(asset_key: str, partition_key: str = "set_01|42"):
    return dg.build_output_context(
        asset_key=dg.AssetKey(asset_key),
        partition_key=partition_key,
    )


def _input_context(asset_key: str, partition_key: str = "set_01|42"):
    return dg.build_input_context(
        asset_key=dg.AssetKey(asset_key),
        partition_key=partition_key,
    )


# ---------------------------------------------------------------------------
# handle_output
# ---------------------------------------------------------------------------


def test_handle_output_creates_sidecar(io_manager, tmp_path):
    ckpt = "/lake/dev/alice/set_01/vgae_small_ae_abc/seed_42/checkpoints/best_model.ckpt"

    io_manager.handle_output(_output_context("ae_abc"), ckpt)

    sidecar = tmp_path / "ae_abc" / "set_01|42.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["checkpoint_path"] == ckpt


def test_handle_output_overwrites_on_rerun(io_manager, tmp_path):
    io_manager.handle_output(_output_context("ae_abc"), "/path/v1.ckpt")
    io_manager.handle_output(_output_context("ae_abc"), "/path/v2.ckpt")

    data = json.loads((tmp_path / "ae_abc" / "set_01|42.json").read_text())
    assert data["checkpoint_path"] == "/path/v2.ckpt"


# ---------------------------------------------------------------------------
# load_input
# ---------------------------------------------------------------------------


def test_load_input_reads_sidecar(io_manager, tmp_path):
    # Arrange: manually write a sidecar
    sidecar_dir = tmp_path / "ae_abc"
    sidecar_dir.mkdir()
    (sidecar_dir / "set_01|42.json").write_text(
        json.dumps({"checkpoint_path": "/lake/best.ckpt"})
    )

    # Act + Assert
    assert io_manager.load_input(_input_context("ae_abc")) == "/lake/best.ckpt"


def test_load_input_missing_sidecar_raises(io_manager):
    with pytest.raises(FileNotFoundError, match="No checkpoint path sidecar"):
        io_manager.load_input(_input_context("nonexistent_asset"))


# ---------------------------------------------------------------------------
# Round-trip and isolation
# ---------------------------------------------------------------------------


def test_round_trip(io_manager):
    ckpt = "/lake/dev/bob/set_02/gat_large_cur_def/seed_0/checkpoints/best_model.ckpt"

    io_manager.handle_output(_output_context("cur_def", "set_02|0"), ckpt)

    assert io_manager.load_input(_input_context("cur_def", "set_02|0")) == ckpt


def test_different_partitions_isolated(io_manager):
    """Each partition gets its own sidecar file."""
    io_manager.handle_output(_output_context("ae_abc", "set_01|42"), "/path/set01.ckpt")
    io_manager.handle_output(_output_context("ae_abc", "set_02|42"), "/path/set02.ckpt")

    assert io_manager.load_input(_input_context("ae_abc", "set_01|42")) == "/path/set01.ckpt"
    assert io_manager.load_input(_input_context("ae_abc", "set_02|42")) == "/path/set02.ckpt"
