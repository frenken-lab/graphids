from __future__ import annotations


def test_build_slurm_script_uses_experiment_resources(monkeypatch, tmp_path):
    from graphids.exp.config import ExperimentConfig, ResourceConfig
    from graphids.exp.slurm import build_slurm_script

    monkeypatch.setenv("GRAPHIDS_SLURM_LOG_DIR", str(tmp_path / "slurm"))
    cfg = ExperimentConfig(
        experiment_name="sequence smoke",
        dataset="set_01",
        resources=ResourceConfig(
            cluster="pitzer",
            partition="gpu",
            accelerator="gpu",
            cpus_per_worker=4,
            gpus_per_worker=1.0,
            time_limit="00:30:00",
            account="pas1266",
        ),
    )
    path, script = build_slurm_script(cfg, "configs/experiments/gat_snapshot_sequence_smoke.yml")

    assert path == tmp_path / "slurm" / "scripts" / "sequence_smoke.sbatch"
    assert "#SBATCH --clusters=pitzer" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --time=00:30:00" in script
    assert "#SBATCH --account=pas1266" in script
    assert "python -m graphids exp launch" in script


def test_submit_experiment_writes_script_and_calls_sbatch(monkeypatch, tmp_path):
    import subprocess

    from graphids.exp.config import ExperimentConfig, ResourceConfig
    from graphids.exp.slurm import submit_experiment

    calls: list[tuple[str, ...]] = []

    def fake_run(command, **kwargs):
        if isinstance(command, str):
            command_tuple = (command,)
        else:
            command_tuple = tuple(command)
        if command_tuple and command_tuple[0] == "sbatch":
            calls.append(command_tuple)
            return subprocess.CompletedProcess(command, 0, stdout="Submitted batch job 12345\n")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setenv("GRAPHIDS_SLURM_LOG_DIR", str(tmp_path / "slurm"))
    monkeypatch.setattr(subprocess, "run", fake_run)
    cfg = ExperimentConfig(
        experiment_name="sequence-smoke",
        dataset="set_01",
        resources=ResourceConfig(accelerator="gpu", gpus_per_worker=1.0, account="pas1266"),
    )

    result = submit_experiment(cfg, "configs/experiments/gat_snapshot_sequence_smoke.yml")

    assert result.job_id == "12345"
    assert result.script_path.is_file()
    assert calls == [("sbatch", str(result.script_path))]


def test_exp_submit_dry_run_cli(monkeypatch):
    from typer.testing import CliRunner

    from graphids.cli.app import app

    monkeypatch.setenv("GRAPHIDS_SLURM_LOG_DIR", "/tmp/graphids-slurm-test")
    result = CliRunner().invoke(
        app,
        ["exp", "submit", "configs/experiments/gat_snapshot_sequence_smoke.yml", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "#SBATCH --clusters=pitzer" in result.stdout
    assert "python -m graphids exp launch" in result.stdout


def test_exp_cache_audit_cli(tmp_path):
    from typer.testing import CliRunner

    from graphids.cli.app import app
    from graphids.core.data.preprocessing.metadata import merge_split_into_metadata
    from tests.core.preprocessing.test_metadata_merge import INVARIANTS, _entry

    audit = {
        "graph_index_overlap": 0,
        "base_unit_overlap": 0,
        "raw_interval_intersections": 0,
        "source_boundary_violations": 0,
    }
    merge_split_into_metadata(
        tmp_path,
        "train",
        {**_entry(10), "split_audit": audit},
        invariants=INVARIANTS,
        dataset_name="set_01",
        num_arb_ids=16,
    )
    merge_split_into_metadata(
        tmp_path,
        "val",
        {"num_graphs": 2, "derived_from": "train", "split_audit": audit},
        invariants=INVARIANTS,
        dataset_name="set_01",
        num_arb_ids=16,
    )

    result = CliRunner().invoke(app, ["exp", "cache-audit", str(tmp_path), "--format", "json"])

    assert result.exit_code == 0
    assert '"ok": true' in result.stdout
