"""Tests for pipes_slurm.py — Dagster Pipes SLURM client + submit_no_poll.

Monkeypatches filesystem paths and SLURM submission to avoid side effects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphids.pipeline.orchestration import pipes_slurm as ps
from graphids.pipeline.orchestration.job import ResourceSpec


@pytest.fixture()
def _isolated_dirs(monkeypatch, tmp_path):
    """Redirect script + pipes dirs to tmp_path so tests don't write to project."""
    monkeypatch.setattr(ps, "_SCRIPTS_DIR", tmp_path / "scripts")
    monkeypatch.setattr(ps, "_PIPES_DIR", tmp_path / "pipes")
    return tmp_path


@pytest.fixture()
def gpu_res():
    from datetime import timedelta

    return ResourceSpec(
        partition="gpu", gpus=1, cpus=4,
        memory_gb=20, walltime=timedelta(hours=3),
    )


# ---------------------------------------------------------------------------
# submit_no_poll
# ---------------------------------------------------------------------------


class TestSubmitNoPoll:
    def test_dry_run_returns_dry_run(self, _isolated_dirs, gpu_res):
        job_id = ps.submit_no_poll(
            stage="autoencoder", model="vgae", scale="large",
            dataset="hcrl_sa", resources=gpu_res, dry_run=True,
        )
        assert job_id == "dry-run"

    def test_dry_run_writes_script_file(self, _isolated_dirs, gpu_res):
        ps.submit_no_poll(
            stage="autoencoder", model="vgae", scale="large",
            dataset="hcrl_sa", resources=gpu_res, dry_run=True,
        )
        scripts = list((_isolated_dirs / "scripts").glob("*.sbatch"))
        assert len(scripts) == 1

    def test_dry_run_script_is_valid_bash(self, _isolated_dirs, gpu_res):
        ps.submit_no_poll(
            stage="autoencoder", model="vgae", scale="large",
            dataset="hcrl_sa", resources=gpu_res, dry_run=True,
        )
        script = list((_isolated_dirs / "scripts").glob("*.sbatch"))[0]
        content = script.read_text()
        assert content.startswith("#!/usr/bin/env bash\n")
        assert "stage=autoencoder" in content

    def test_dependency_in_script(self, _isolated_dirs, gpu_res):
        ps.submit_no_poll(
            stage="curriculum", model="gat", scale="large",
            dataset="hcrl_sa", resources=gpu_res,
            dependency_job_id="12345", dry_run=True,
        )
        script = list((_isolated_dirs / "scripts").glob("*.sbatch"))[0]
        assert "#SBATCH --dependency=afterok:12345" in script.read_text()

    def test_real_submit_calls_sbatch(self, _isolated_dirs, gpu_res, monkeypatch):
        """When dry_run=False, submit_sbatch is called."""
        submitted = []
        monkeypatch.setattr(
            ps, "submit_sbatch",
            lambda path, **kw: (submitted.append(path), "99999")[1],
        )
        job_id = ps.submit_no_poll(
            stage="autoencoder", model="vgae", scale="large",
            dataset="hcrl_sa", resources=gpu_res, dry_run=False,
        )
        assert job_id == "99999"
        assert len(submitted) == 1
        assert Path(submitted[0]).suffix == ".sbatch"


# ---------------------------------------------------------------------------
# CLI subcommand registration
# ---------------------------------------------------------------------------


class TestCLI:
    def test_orchestrate_is_subcommand(self):
        from graphids.cli import _SUBCOMMANDS

        assert "orchestrate" in _SUBCOMMANDS

    def test_preprocess_is_subcommand(self):
        from graphids.cli import _SUBCOMMANDS

        assert "preprocess" in _SUBCOMMANDS
