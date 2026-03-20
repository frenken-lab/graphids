"""Tests for slurm_primitives.py.

Covers resource profiles, adaptive retry scaling, sbatch script generation,
sacct output parsing, and script file writing.
"""

from __future__ import annotations

import subprocess
from datetime import timedelta
from types import SimpleNamespace

import pytest

from graphids.pipeline.orchestration.job import ResourceSpec
from graphids.pipeline.orchestration import slurm_primitives as sp


# ---------------------------------------------------------------------------
# Resource profiles (loaded from resources.yaml at import time)
# ---------------------------------------------------------------------------


class TestResourceProfiles:
    def test_profiles_not_empty(self):
        assert len(sp.RESOURCE_PROFILES) > 0

    def test_all_values_are_resource_specs(self):
        for key, res in sp.RESOURCE_PROFILES.items():
            assert isinstance(res, ResourceSpec), f"{key} is {type(res)}"

    @pytest.mark.parametrize(
        ("model", "scale", "stage", "expected_partition", "expected_gpus"),
        [
            ("vgae", "large", "autoencoder", "gpu", 1),
            ("preprocess", "", "preprocess", "cpu", 0),
            ("dqn", "large", "fusion", "cpu", 0),
        ],
        ids=["vgae-gpu", "preprocess-cpu", "dqn-cpu"],
    )
    def test_known_profiles(self, model, scale, stage, expected_partition, expected_gpus):
        res = sp.get_resources(model, scale, stage)
        assert res.partition == expected_partition
        assert res.gpus == expected_gpus

    def test_missing_profile_raises_keyerror(self):
        with pytest.raises(KeyError, match="No resource profile"):
            sp.get_resources("nonexistent", "large", "autoencoder")


# ---------------------------------------------------------------------------
# Failure reactions
# ---------------------------------------------------------------------------


class TestFailureReactions:
    def test_expected_reasons_present(self):
        for reason in ("OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL"):
            assert reason in sp.FAILURE_REACTIONS

    def test_oom_scale_factor(self):
        assert sp.FAILURE_REACTIONS["OUT_OF_MEMORY"]["scale_mem"] == 2.0

    def test_timeout_enables_ckpt_resume(self):
        assert sp.FAILURE_REACTIONS["TIMEOUT"]["ckpt_resume"] is True


# ---------------------------------------------------------------------------
# Adaptive retry scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "base_mem", "base_hours", "expected_mem", "expected_secs"),
    [
        ("OUT_OF_MEMORY", 16, 2, 32, 7200),       # 2x mem, time unchanged
        ("TIMEOUT", 16, 2, 16, 10800),             # mem unchanged, 1.5x time
        ("NODE_FAIL", 16, 2, 16, 7200),            # nothing changes
        ("UNKNOWN_REASON", 16, 2, 16, 7200),       # nothing changes
    ],
    ids=["oom-doubles-mem", "timeout-scales-time", "node-fail-noop", "unknown-noop"],
)
def test_scale_resources(reason, base_mem, base_hours, expected_mem, expected_secs):
    base = ResourceSpec(
        partition="gpu", gpus=1, cpus=4,
        memory_gb=base_mem, walltime=timedelta(hours=base_hours),
    )
    scaled = sp.scale_resources(base, reason)
    assert scaled.memory_gb == expected_mem
    assert int(scaled.walltime.total_seconds()) == expected_secs


# ---------------------------------------------------------------------------
# Sbatch script generation
# ---------------------------------------------------------------------------


class TestGenerateSbatchScript:
    """Each test builds a script and checks specific directives."""

    @pytest.fixture()
    def gpu_script(self, gpu_resources):
        return sp.generate_sbatch_script(
            stage="autoencoder", model="vgae", scale="large",
            dataset="hcrl_sa", resources=gpu_resources,
        )

    @pytest.fixture()
    def cpu_script(self, cpu_resources):
        return sp.generate_sbatch_script(
            stage="preprocess", model="preprocess", scale="",
            dataset="hcrl_sa", resources=cpu_resources,
        )

    # --- structural ---

    def test_shebang(self, gpu_script):
        assert gpu_script.startswith("#!/usr/bin/env bash\n")

    def test_preamble_and_epilog(self, gpu_script):
        assert "source scripts/slurm/_preamble.sh" in gpu_script
        assert "source scripts/slurm/_epilog.sh" in gpu_script

    def test_background_and_wait_pattern(self, gpu_script):
        assert "&\n" in gpu_script
        assert "_KD_CHILD_PID=$!" in gpu_script
        assert "wait $_KD_CHILD_PID" in gpu_script
        assert "exit $EXIT_CODE" in gpu_script

    # --- GPU vs CPU ---

    def test_gpu_has_gres(self, gpu_script):
        assert "--gres=gpu:" in gpu_script

    def test_gpu_no_skip_cuda(self, gpu_script):
        assert "SKIP_CUDA_CONF" not in gpu_script

    def test_cpu_no_gres(self, cpu_script):
        assert "--gres=gpu" not in cpu_script

    def test_cpu_skips_cuda(self, cpu_script):
        assert "SKIP_CUDA_CONF=1" in cpu_script

    # --- SLURM directives ---

    def test_partition(self, gpu_script):
        assert "#SBATCH --partition=gpu" in gpu_script

    def test_memory(self, gpu_script):
        assert "#SBATCH --mem=16G" in gpu_script

    def test_walltime(self, gpu_script):
        assert "#SBATCH --time=3:00:00" in gpu_script

    def test_job_name(self, gpu_script):
        assert "#SBATCH --job-name=kd-gat-autoencoder-vgae-large" in gpu_script

    def test_signal(self, gpu_script):
        assert "#SBATCH --signal=B:USR1@180" in gpu_script

    # --- CLI command ---

    def test_cli_module(self, gpu_script):
        assert "graphids.cli" in gpu_script

    def test_stage_in_command(self, gpu_script):
        assert "stage=autoencoder" in gpu_script

    def test_model_in_command(self, gpu_script):
        assert "model=vgae_large" in gpu_script

    def test_dataset_in_command(self, gpu_script):
        assert "dataset=hcrl_sa" in gpu_script

    # --- optional parameters ---

    @pytest.mark.parametrize(
        ("kwargs", "expected_fragment"),
        [
            ({"dependency_job_id": "12345"}, "#SBATCH --dependency=afterok:12345"),
            ({"seed": 42}, "seed=42"),
            ({"auxiliaries": "kd_standard"}, "auxiliary=kd_standard"),
            ({"ckpt_path": "/tmp/ckpt.pt"}, 'KD_GAT_CKPT_PATH="/tmp/ckpt.pt"'),
            ({"extra_env": {"DAGSTER_PIPES_CONTEXT": "/tmp/ctx"}}, 'export DAGSTER_PIPES_CONTEXT="/tmp/ctx"'),
        ],
        ids=["dependency", "seed", "auxiliaries", "ckpt-path", "extra-env"],
    )
    def test_optional_params(self, gpu_resources, kwargs, expected_fragment):
        script = sp.generate_sbatch_script(
            stage="autoencoder", model="vgae", scale="large",
            dataset="hcrl_sa", resources=gpu_resources, **kwargs,
        )
        assert expected_fragment in script

    def test_exclude_nodes(self):
        res = ResourceSpec(
            partition="gpu", gpus=1, cpus=4, memory_gb=16,
            walltime=timedelta(hours=3), exclude_nodes="p0042",
        )
        script = sp.generate_sbatch_script(
            stage="autoencoder", model="vgae", scale="large",
            dataset="hcrl_sa", resources=res,
        )
        assert "#SBATCH --exclude=p0042" in script


# ---------------------------------------------------------------------------
# sacct output parsing
# ---------------------------------------------------------------------------


class TestSacctQuery:
    @pytest.mark.parametrize(
        ("stdout", "returncode", "expected_state"),
        [
            ("COMPLETED|None|p0042\n", 0, "COMPLETED"),
            ("FAILED|OOM|p0042\n", 0, "FAILED"),
            ("RUNNING |None|p0042\n", 0, "RUNNING"),  # trailing space
            ("", 0, "PENDING"),            # empty output
            ("something", 1, "PENDING"),   # nonzero return code
        ],
        ids=["completed", "failed", "running-whitespace", "empty", "error-rc"],
    )
    def test_sacct_parsing(self, monkeypatch, stdout, returncode, expected_state):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: SimpleNamespace(returncode=returncode, stdout=stdout),
        )
        state, _reason, _node = sp.sacct_query("12345")
        assert state == expected_state

    def test_sacct_extracts_reason_and_node(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stdout="FAILED|OOM kill|p0042\n"),
        )
        state, reason, node = sp.sacct_query("12345")
        assert state == "FAILED"
        assert reason == "OOM kill"
        assert node == "p0042"


# ---------------------------------------------------------------------------
# poll_until_done
# ---------------------------------------------------------------------------


def test_poll_until_done_returns_on_terminal(monkeypatch):
    """Monkeypatch sacct_query to return PENDING twice then COMPLETED."""
    call_count = 0

    def fake_sacct(job_id):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return "PENDING", "", ""
        return "COMPLETED", "None", "p0042"

    monkeypatch.setattr(sp, "sacct_query", fake_sacct)
    monkeypatch.setattr(sp.time, "sleep", lambda _: None)  # skip real sleep

    state, reason, node = sp.poll_until_done("12345", poll_interval=1)
    assert state == "COMPLETED"
    assert node == "p0042"
    assert call_count == 3


# ---------------------------------------------------------------------------
# write_script_file
# ---------------------------------------------------------------------------


class TestWriteScriptFile:
    def test_writes_content_to_disk(self, tmp_path):
        content = "#!/bin/bash\necho hello\n"
        path = sp.write_script_file(content, tmp_path, "vgae", "large", "autoencoder")
        assert path.exists()
        assert path.read_text() == content

    def test_filename_format(self, tmp_path):
        path = sp.write_script_file("x", tmp_path, "vgae", "large", "autoencoder")
        assert path.name == "dagster_vgae_large_autoencoder.sbatch"

    def test_auxiliaries_in_filename(self, tmp_path):
        path = sp.write_script_file("x", tmp_path, "gat", "small", "curriculum", "kd_standard")
        assert path.name == "dagster_gat_small_curriculum_kd_standard.sbatch"

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b"
        path = sp.write_script_file("x", nested, "vgae", "large", "autoencoder")
        assert path.exists()


# ---------------------------------------------------------------------------
# SlurmJobFailed exception
# ---------------------------------------------------------------------------


class TestSlurmJobFailed:
    def test_attributes(self):
        exc = sp.SlurmJobFailed(reason="TIMEOUT", node="p0042", ckpt_path="/tmp/ckpt")
        assert exc.reason == "TIMEOUT"
        assert exc.node == "p0042"
        assert exc.ckpt_path == "/tmp/ckpt"

    def test_str_includes_reason(self):
        exc = sp.SlurmJobFailed(reason="OUT_OF_MEMORY", node="p0001")
        assert "OUT_OF_MEMORY" in str(exc)

    def test_metadata_defaults_to_empty(self):
        exc = sp.SlurmJobFailed(reason="FAILED")
        assert exc.metadata == {}
