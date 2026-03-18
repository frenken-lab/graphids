"""Tests for Dagster SLURM orchestration (Phase 1).

Unit tests that run on the login node without SLURM submission.
Tests script generation, resource loading, resource scaling,
dry-run mode, and DAG construction.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from graphids.pipeline.orchestration.job import ResourceSpec
from graphids.pipeline.orchestration.pipes_slurm import (
    FAILURE_REACTIONS,
    RESOURCE_PROFILES,
    PipesSlurmClient,
    generate_sbatch_script,
    get_resources,
    scale_resources,
)


def _gpu(*, memory_gb: int = 16, hours: int = 3, **kw) -> ResourceSpec:
    """Shorthand for GPU resource spec."""
    return ResourceSpec(
        partition="gpu",
        gpus=1,
        cpus=4,
        memory_gb=memory_gb,
        walltime=timedelta(hours=hours),
        **kw,
    )


def _cpu(*, memory_gb: int = 32, hours: int = 1, **kw) -> ResourceSpec:
    """Shorthand for CPU resource spec."""
    return ResourceSpec(
        partition="serial",
        gpus=0,
        cpus=8,
        memory_gb=memory_gb,
        walltime=timedelta(hours=hours),
        **kw,
    )


# ---------------------------------------------------------------------------
# ResourceSpec SLURM properties + from_yaml
# ---------------------------------------------------------------------------


class TestResourceSpec:
    def test_mem_slurm(self):
        assert ResourceSpec(memory_gb=20).mem_slurm == "20G"

    def test_walltime_slurm_hours(self):
        assert ResourceSpec(walltime=timedelta(hours=3)).walltime_slurm == "3:00:00"

    def test_walltime_slurm_half_hour(self):
        assert ResourceSpec(walltime=timedelta(minutes=30)).walltime_slurm == "0:30:00"

    def test_walltime_slurm_mixed(self):
        assert ResourceSpec(walltime=timedelta(hours=1, minutes=30)).walltime_slurm == "1:30:00"

    def test_frozen(self):
        res = _gpu()
        with pytest.raises(Exception):
            res.gpus = 2  # type: ignore[misc]

    def test_default_exclude_nodes(self):
        assert ResourceSpec().exclude_nodes == ""

    def test_from_yaml_with_mem_string(self):
        res = ResourceSpec.from_yaml(
            {
                "partition": "gpu",
                "gpus": 1,
                "cpus": 4,
                "mem": "20G",
                "walltime": "3:00:00",
            }
        )
        assert res.memory_gb == 20
        assert res.walltime == timedelta(hours=3)
        assert res.partition == "gpu"

    def test_from_yaml_with_memory_gb(self):
        res = ResourceSpec.from_yaml(
            {
                "partition": "serial",
                "gpus": 0,
                "cpus": 8,
                "memory_gb": 32,
                "walltime": "1:00:00",
            }
        )
        assert res.memory_gb == 32

    def test_from_yaml_defaults(self):
        res = ResourceSpec.from_yaml({})
        assert res.memory_gb == 20
        assert res.partition == "cpu"


# ---------------------------------------------------------------------------
# Resource profile loading
# ---------------------------------------------------------------------------


class TestResourceProfiles:
    def test_profiles_loaded(self):
        """resources.yaml is parsed into RESOURCE_PROFILES at import time."""
        assert len(RESOURCE_PROFILES) > 0

    def test_vgae_large_autoencoder_exists(self):
        res = get_resources("vgae", "large", "autoencoder")
        assert res.partition == "gpu"
        assert res.gpus == 1
        assert res.memory_gb == 20

    def test_preprocess_is_cpu(self):
        res = get_resources("preprocess", "", "preprocess")
        assert res.gpus == 0
        assert res.partition == "cpu"

    def test_dqn_fusion_is_cpu(self):
        res = get_resources("dqn", "large", "fusion")
        assert res.gpus == 0
        assert res.partition == "cpu"

    def test_missing_profile_raises(self):
        with pytest.raises(KeyError, match="No resource profile"):
            get_resources("nonexistent", "large", "autoencoder")

    def test_all_profiles_are_resource_specs(self):
        for key, res in RESOURCE_PROFILES.items():
            assert isinstance(res, ResourceSpec), f"{key}: not ResourceSpec"


# ---------------------------------------------------------------------------
# Failure reactions loading
# ---------------------------------------------------------------------------


class TestFailureReactions:
    def test_reactions_loaded(self):
        assert "OUT_OF_MEMORY" in FAILURE_REACTIONS
        assert "TIMEOUT" in FAILURE_REACTIONS
        assert "NODE_FAIL" in FAILURE_REACTIONS

    def test_oom_has_scale_mem(self):
        assert FAILURE_REACTIONS["OUT_OF_MEMORY"]["scale_mem"] == 2.0

    def test_timeout_has_ckpt_resume(self):
        assert FAILURE_REACTIONS["TIMEOUT"]["ckpt_resume"] is True


# ---------------------------------------------------------------------------
# Resource scaling
# ---------------------------------------------------------------------------


class TestResourceScaling:
    def test_oom_doubles_memory(self):
        base = _gpu(memory_gb=16)
        scaled = scale_resources(base, "OUT_OF_MEMORY")
        assert scaled.memory_gb == 32
        assert scaled.walltime == base.walltime  # unchanged

    def test_timeout_scales_time(self):
        base = _gpu(memory_gb=16, hours=2)
        scaled = scale_resources(base, "TIMEOUT")
        assert scaled.walltime == timedelta(seconds=10800)  # 7200 * 1.5
        assert scaled.memory_gb == 16  # unchanged

    def test_unknown_reason_no_change(self):
        base = _gpu(memory_gb=16, hours=2)
        scaled = scale_resources(base, "UNKNOWN_REASON")
        assert scaled == base

    def test_node_fail_preserves_resources(self):
        base = _gpu(memory_gb=16, hours=2)
        scaled = scale_resources(base, "NODE_FAIL")
        assert scaled == base


# ---------------------------------------------------------------------------
# Sbatch script generation
# ---------------------------------------------------------------------------


class TestScriptGeneration:
    def test_basic_gpu_script(self):
        res = _gpu(memory_gb=20)
        script = generate_sbatch_script(
            stage="autoencoder",
            model="vgae",
            scale="large",
            dataset="hcrl_sa",
            resources=res,
        )
        assert "#!/usr/bin/env bash" in script
        assert "#SBATCH --partition=gpu" in script
        assert "--gres=gpu:" in script
        assert "#SBATCH --mem=20G" in script
        assert "#SBATCH --time=3:00:00" in script
        assert "#SBATCH --job-name=kd-gat-autoencoder-vgae-large" in script
        assert "source scripts/slurm/_preamble.sh" in script
        assert "source scripts/slurm/_epilog.sh" in script
        assert "graphids.pipeline.cli" in script
        assert "stage=autoencoder" in script
        assert "model=vgae_large" in script
        assert "dataset=hcrl_sa" in script
        assert "SKIP_CUDA_CONF" not in script  # GPU job

    def test_cpu_script_skips_cuda_and_gpu(self):
        res = _cpu()
        script = generate_sbatch_script(
            stage="preprocess",
            model="preprocess",
            scale="",
            dataset="hcrl_sa",
            resources=res,
        )
        assert "--gres=gpu" not in script
        assert "SKIP_CUDA_CONF=1" in script

    def test_dependency_flag(self):
        res = _gpu()
        script = generate_sbatch_script(
            stage="curriculum",
            model="gat",
            scale="large",
            dataset="hcrl_sa",
            resources=res,
            dependency_job_id="12345",
        )
        assert "#SBATCH --dependency=afterok:12345" in script

    def test_exclude_nodes(self):
        res = _gpu(exclude_nodes="p0042")
        script = generate_sbatch_script(
            stage="autoencoder",
            model="vgae",
            scale="large",
            dataset="hcrl_sa",
            resources=res,
        )
        assert "#SBATCH --exclude=p0042" in script

    def test_auxiliaries_in_command(self):
        res = _gpu(memory_gb=12, hours=2)
        script = generate_sbatch_script(
            stage="curriculum",
            model="gat",
            scale="small",
            dataset="hcrl_sa",
            resources=res,
            auxiliaries="kd_standard",
        )
        assert "auxiliary=kd_standard" in script

    def test_ckpt_path_in_command(self):
        res = _gpu()
        script = generate_sbatch_script(
            stage="autoencoder",
            model="vgae",
            scale="large",
            dataset="hcrl_sa",
            resources=res,
            ckpt_path="/tmp/ckpt.pt",
        )
        assert 'KD_GAT_CKPT_PATH="/tmp/ckpt.pt"' in script

    def test_seed_in_command(self):
        res = _gpu()
        script = generate_sbatch_script(
            stage="autoencoder",
            model="vgae",
            scale="large",
            dataset="hcrl_sa",
            resources=res,
            seed=42,
        )
        assert "seed=42" in script

    def test_sigusr1_signal(self):
        res = _gpu()
        script = generate_sbatch_script(
            stage="autoencoder",
            model="vgae",
            scale="large",
            dataset="hcrl_sa",
            resources=res,
        )
        assert "#SBATCH --signal=B:USR1@180" in script

    def test_child_pid_wait_pattern(self):
        """Ensure the script backgrounds the command and waits for it."""
        res = _gpu()
        script = generate_sbatch_script(
            stage="autoencoder",
            model="vgae",
            scale="large",
            dataset="hcrl_sa",
            resources=res,
        )
        assert "&\n" in script
        assert "_KD_CHILD_PID=$!" in script
        assert "wait $_KD_CHILD_PID" in script


# ---------------------------------------------------------------------------
# Retry state helpers
# ---------------------------------------------------------------------------


class TestRetryState:
    def test_save_load_clear(self, tmp_path, monkeypatch):
        from graphids.pipeline.orchestration import dagster_resources as dr

        monkeypatch.setattr(dr, "_RETRY_STATE_DIR", tmp_path / "retry")

        dr.save_retry_state("test_asset", "OUT_OF_MEMORY", node="p0042")
        state = dr.load_retry_state("test_asset")
        assert state is not None
        assert state["reason"] == "OUT_OF_MEMORY"
        assert state["node"] == "p0042"

        dr.clear_retry_state("test_asset")
        assert dr.load_retry_state("test_asset") is None

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        from graphids.pipeline.orchestration import dagster_resources as dr

        monkeypatch.setattr(dr, "_RETRY_STATE_DIR", tmp_path / "retry")
        assert dr.load_retry_state("nonexistent") is None


# ---------------------------------------------------------------------------
# PipesSlurmClient dry-run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_returns_metadata(self, tmp_path):
        client = PipesSlurmClient(project_root=str(tmp_path), dry_run=True)
        (tmp_path / "slurm_logs").mkdir()

        result = client.run(
            stage="autoencoder",
            model="vgae",
            scale="large",
            dataset="hcrl_sa",
            resources=_gpu(memory_gb=20),
        )
        assert result["job_id"] == "dry-run"
        assert result["state"] == "DRY_RUN"
        assert "script_path" in result

    def test_dry_run_writes_script_file(self, tmp_path):
        client = PipesSlurmClient(project_root=str(tmp_path), dry_run=True)
        (tmp_path / "slurm_logs").mkdir()

        result = client.run(
            stage="autoencoder",
            model="vgae",
            scale="large",
            dataset="hcrl_sa",
            resources=_gpu(memory_gb=20),
        )
        script_path = result["script_path"]
        assert Path(script_path).exists()
        assert "#!/usr/bin/env bash" in Path(script_path).read_text()


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------


def _get_assets():
    from graphids.pipeline.orchestration.dagster_defs import build_dagster_assets

    return build_dagster_assets()


def _dep_names(asset) -> set[str]:
    """Extract dependency asset key names from an asset."""
    spec = list(asset.specs)[0]
    return {d.asset_key.path[-1] for d in spec.deps}


def _find(assets, name):
    return next(a for a in assets if a.key.path[-1] == name)


class TestDAGStructure:
    def test_total_asset_count(self):
        """3 variants × 4 stages + preprocess + hf_push + rebuild_catalog = 15."""
        assets = _get_assets()
        assert len(assets) == 15

    def test_all_expected_assets_present(self):
        assets = _get_assets()
        names = {a.key.path[-1] for a in assets}
        expected = {
            "preprocess",
            # Large variant
            "vgae_large_autoencoder",
            "gat_large_curriculum",
            "dqn_large_fusion",
            "eval_large_evaluation",
            # Small KD variant
            "vgae_small_autoencoder_kd_standard",
            "gat_small_curriculum_kd_standard",
            "dqn_small_fusion_kd_standard",
            "eval_small_evaluation_kd_standard",
            # Small no-KD variant
            "vgae_small_autoencoder",
            "gat_small_curriculum",
            "dqn_small_fusion",
            "eval_small_evaluation",
            # Final
            "hf_push",
            "rebuild_catalog",
        }
        assert names == expected

    def test_preprocess_has_no_deps(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "preprocess")) == set()


class TestLargeVariantDeps:
    def test_vgae_depends_on_preprocess(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "vgae_large_autoencoder")) == {"preprocess"}

    def test_gat_depends_on_vgae(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "gat_large_curriculum")) == {"vgae_large_autoencoder"}

    def test_dqn_depends_on_vgae_and_gat(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "dqn_large_fusion")) == {
            "vgae_large_autoencoder",
            "gat_large_curriculum",
        }

    def test_eval_depends_on_dqn(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "eval_large_evaluation")) == {"dqn_large_fusion"}


class TestSmallKdVariantDeps:
    """Small KD variant: needs_teacher=True → cross-variant deps to large."""

    def test_vgae_kd_depends_on_preprocess_and_teacher(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "vgae_small_autoencoder_kd_standard")) == {
            "preprocess",
            "vgae_large_autoencoder",  # teacher
        }

    def test_gat_kd_depends_on_own_vgae_and_teacher(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "gat_small_curriculum_kd_standard")) == {
            "vgae_small_autoencoder_kd_standard",  # intra-variant
            "gat_large_curriculum",  # teacher
        }

    def test_dqn_kd_has_no_teacher_dep(self):
        """DQN fusion doesn't use KD — no cross-variant dep."""
        assets = _get_assets()
        assert _dep_names(_find(assets, "dqn_small_fusion_kd_standard")) == {
            "vgae_small_autoencoder_kd_standard",
            "gat_small_curriculum_kd_standard",
        }

    def test_eval_kd_depends_on_dqn(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "eval_small_evaluation_kd_standard")) == {
            "dqn_small_fusion_kd_standard"
        }


class TestSmallNokdVariantDeps:
    """Small no-KD variant: independent, no teacher deps."""

    def test_vgae_nokd_depends_only_on_preprocess(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "vgae_small_autoencoder")) == {"preprocess"}

    def test_gat_nokd_depends_on_own_vgae(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "gat_small_curriculum")) == {"vgae_small_autoencoder"}

    def test_eval_nokd_depends_on_dqn(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "eval_small_evaluation")) == {"dqn_small_fusion"}


class TestHfPush:
    def test_hf_push_depends_on_all_evals(self):
        assets = _get_assets()
        assert _dep_names(_find(assets, "hf_push")) == {
            "eval_large_evaluation",
            "eval_small_evaluation_kd_standard",
            "eval_small_evaluation",
        }


class TestAssetMetadata:
    def test_eval_uses_vgae_cli_model(self):
        """Evaluation CLI gets --model vgae, not --model eval."""
        assets = _get_assets()
        spec = list(_find(assets, "eval_large_evaluation").specs)[0]
        assert spec.metadata["cli_model"] == "vgae"
        assert spec.metadata["resource_model"] == "eval"

    def test_all_assets_have_multi_partitions(self):
        """Every asset should have MultiPartitionsDefinition(dataset, seed)."""
        import dagster as dg

        assets = _get_assets()
        for a in assets:
            spec = list(a.specs)[0]
            assert isinstance(spec.partitions_def, dg.MultiPartitionsDefinition), (
                f"{a.key.path[-1]} missing multi-partitions"
            )


# ---------------------------------------------------------------------------
# Fire-and-forget mode (Phase 3)
# ---------------------------------------------------------------------------


class TestFireAndForget:
    def test_dry_run_returns_all_jobs(self):
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        job_ids = fire_and_forget(dataset="hcrl_sa", dry_run=True)
        # 13 stages (preprocess + 4×3 variants), all dry-run
        assert len(job_ids) == 13
        assert all(v == "dry-run" for v in job_ids.values())

    def test_dry_run_includes_all_variants(self):
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        job_ids = fire_and_forget(dataset="hcrl_sa", dry_run=True)
        names = set(job_ids.keys())
        # Check key assets are present (with seed suffix)
        assert any("preprocess" in n for n in names)
        assert any("vgae_large_autoencoder" in n for n in names)
        assert any("eval_small_evaluation_kd_standard" in n for n in names)
        assert any("gat_small_curriculum__seed42" in n for n in names)

    def test_dry_run_multi_seed(self):
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        job_ids = fire_and_forget(dataset="hcrl_sa", seeds=[42, 123], dry_run=True)
        # 13 stages × 2 seeds = 26
        assert len(job_ids) == 26
        assert any("seed42" in k for k in job_ids)
        assert any("seed123" in k for k in job_ids)

    def test_topological_order(self):
        """Preprocess is submitted before any training stage."""
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        # dry_run returns all "dry-run", but we can verify the function
        # completes without error (topological sort succeeded)
        job_ids = fire_and_forget(dataset="hcrl_sa", dry_run=True)
        keys = list(job_ids.keys())
        # Preprocess should be first
        assert "preprocess__seed42" == keys[0]


# ---------------------------------------------------------------------------
# submit_no_poll (Phase 3)
# ---------------------------------------------------------------------------


class TestSubmitNoPoll:
    def test_dry_run_returns_dry_run_id(self, tmp_path):
        client = PipesSlurmClient(project_root=str(tmp_path), dry_run=True)
        (tmp_path / "slurm_logs").mkdir()

        job_id = client.submit_no_poll(
            stage="autoencoder",
            model="vgae",
            scale="large",
            dataset="hcrl_sa",
            resources=_gpu(memory_gb=20),
        )
        assert job_id == "dry-run"

    def test_dry_run_writes_script(self, tmp_path):
        client = PipesSlurmClient(project_root=str(tmp_path), dry_run=True)
        (tmp_path / "slurm_logs").mkdir()

        client.submit_no_poll(
            stage="autoencoder",
            model="vgae",
            scale="large",
            dataset="hcrl_sa",
            resources=_gpu(memory_gb=20),
        )
        scripts = list((tmp_path / "slurm_logs").glob("*.sbatch"))
        assert len(scripts) == 1

    def test_dependency_in_script(self, tmp_path):
        client = PipesSlurmClient(project_root=str(tmp_path), dry_run=True)
        (tmp_path / "slurm_logs").mkdir()

        client.submit_no_poll(
            stage="curriculum",
            model="gat",
            scale="large",
            dataset="hcrl_sa",
            resources=_gpu(),
            dependency_job_id="12345",
        )
        script = list((tmp_path / "slurm_logs").glob("*.sbatch"))[0].read_text()
        assert "#SBATCH --dependency=afterok:12345" in script


# ---------------------------------------------------------------------------
# CLI orchestrate subcommand (Phase 3)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_orchestrate_is_subcommand(self):
        from graphids.pipeline.cli import _SUBCOMMANDS

        assert "orchestrate" in _SUBCOMMANDS

    def test_preprocess_is_subcommand(self):
        from graphids.pipeline.cli import _SUBCOMMANDS

        assert "preprocess" in _SUBCOMMANDS

    def test_flow_not_a_subcommand(self):
        from graphids.pipeline.cli import _SUBCOMMANDS

        assert "flow" not in _SUBCOMMANDS
