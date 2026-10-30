"""Frozen source, training-only weighting, and scheduler dependency contracts."""

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from graphids.data.recordings import CANRecordingManifest, Recording, file_hash
from scripts import event_quality_suite as quality


def source_fixture(tmp_path):
    records = []
    cache_root = tmp_path / "cache"
    for key, role, normal, labels in (
        ("attack", "train_query", False, [0, 1, 0]),
        ("normal", "train_query", True, [0, 0, 0, 0]),
        ("selection", "val_query", False, [1, 1, 0]),
    ):
        path = tmp_path / f"{key}.csv"
        path.write_text(key)
        rec = Recording(
            key=key, dataset="fixture", vehicle="source", session=key,
            path=str(path), sha256=file_hash(path), role=role,
            verified_normal=normal,
        )
        records.append(rec)
        digest = hashlib.sha256(
            json.dumps(asdict(rec), sort_keys=True).encode()
        ).hexdigest()
        cache = cache_root / f"event-v1-{digest}"
        cache.mkdir(parents=True)
        (cache / "metadata.json").write_text(json.dumps(dict(
            digest=digest, count=len(labels), recording=asdict(rec), schema_version=1
        )))
        np.save(cache / "labels.npy", np.asarray(labels, dtype=np.int64))
        np.save(cache / "ids.npy", np.arange(len(labels), dtype=np.int64))
    # Metadata is allowed; opening this reserved target path would fail.
    records.append(Recording(
        key="reserved", dataset="fixture", vehicle="target", session="reserved",
        path=str(tmp_path / "target-must-not-be-opened.csv"), sha256="f" * 64,
        role="target_attack_test",
    ))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(asdict(CANRecordingManifest(
        tuple(records), provenance_reviewed=True
    ))))
    base = yaml.safe_load(Path(
        "configs/experiments/graph_mamba_events_source.yml"
    ).read_text())
    base["config"]["data"].update(
        manifest_path=str(manifest), cache_root=str(cache_root), datasets=["fixture"]
    )
    return base


def test_weight_is_training_only_and_reserved_targets_are_never_opened(tmp_path):
    result = quality.cohort(source_fixture(tmp_path))
    assert result["training_rows"] == 7
    assert result["validation_rows"] == 3
    assert result["positive_weight"] == 6.0
    assert {r["recording"] for r in result["recordings"]} == {
        "attack", "normal", "selection"
    }


def prepared_suite(tmp_path, monkeypatch):
    base = source_fixture(tmp_path)
    preset = tmp_path / "base.yml"
    preset.write_text(yaml.safe_dump(base))
    repo = tmp_path / "repo"
    (repo / "graphids").mkdir(parents=True)
    (repo / "graphids/runtime.py").write_text("version = 1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run([
        "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "fixture",
    ], cwd=repo, check=True)
    # Dirty and untracked runtime files must both reach the source snapshot.
    (repo / "graphids/runtime.py").write_text("version = 2\n")
    (repo / "graphids/untracked.py").write_text("new_feature = True\n")
    monkeypatch.setattr(quality, "PROJECT_ROOT", repo)
    root = tmp_path / "suite"
    suite = quality.prepare(root, preset, seed=43)
    return root, suite, repo


def test_preparation_matches_budgets_reuses_baseline_and_freezes_dirty_source(
    tmp_path, monkeypatch
):
    root, suite, repo = prepared_suite(tmp_path, monkeypatch)
    jobs = suite["training_jobs"]
    assert len(jobs) == 11
    assert sum("controls" in j["phases"] for j in jobs) == 5
    assert sum("context" in j["phases"] for j in jobs) == 7
    for job in jobs:
        cfg = yaml.safe_load(Path(job["config"]).read_text())
        assert cfg["seed"] == 43
        assert cfg["config"]["trainer"]["max_epochs"] == 25
        assert cfg["config"]["data"]["effective_batch_events"] == 65536
        assert cfg["config"]["model"]["max_graph_tokens"] == 8388608
        script = Path(job["script"]).read_text()
        assert str(root / "source") in script
        assert str(repo) not in script
        assert "scripts.event_quality_suite verify" in script
    assert (root / "source/graphids/runtime.py").read_text() == "version = 2\n"
    assert (root / "source/graphids/untracked.py").exists()
    from scripts.event_model_gate import source_provenance

    git_head, diff, hashes = source_provenance(root / "source")
    assert git_head == suite["git_sha"]
    assert diff == (root / "working-tree.diff").read_bytes()
    assert "graphids/untracked.py" in hashes
    # Subsequent work in the live checkout cannot alter the frozen inputs.
    (repo / "graphids/runtime.py").write_text("version = 3\n")
    quality.verify(root)
    (root / "source/graphids/runtime.py").write_text("version = 4\n")
    with pytest.raises(ValueError, match="frozen suite file changed"):
        quality.verify(root)


def test_submission_blocks_context_on_complete_controls_and_keeps_job_receipts(
    tmp_path, monkeypatch
):
    root, _, _ = prepared_suite(tmp_path, monkeypatch)
    commands = []

    def sbatch(command, **kwargs):
        commands.append(command)
        return f"{100 + len(commands)};ascend\n"

    monkeypatch.setattr(quality.subprocess, "check_output", sbatch)
    suite = quality.submit(root)
    assert suite["status"] == "submitted"
    assert len(commands) == 15  # Two gates, eleven fits, two reports.
    assert all("--kill-on-invalid-dep=yes" in c for c in commands)
    control_gate, context_gate = suite["gates"]
    assert f"--dependency=afterok:{control_gate['job_id']}" in commands[1]
    control_jobs = [j for j in suite["training_jobs"] if "controls" in j["phases"]]
    for job in control_jobs:
        assert f"--dependency=afterok:{control_gate['job_id']}" in job["submit_command"]
    control_report = suite["reports"][0]
    assert "--dependency=afterany:" + ":".join(
        j["job_id"] for j in control_jobs
    ) in control_report["submit_command"]
    assert "--require-complete" in Path(control_report["script"]).read_text()
    for job in suite["training_jobs"]:
        if job["phases"] == ["context"]:
            assert (f"--dependency=afterok:{context_gate['job_id']}:"
                    f"{control_report['job_id']}") in job["submit_command"]
    with pytest.raises(ValueError, match="already submitted"):
        quality.submit(root)


def test_partial_submission_preserves_accepted_jobs_and_prevents_duplicate_retry(
    tmp_path, monkeypatch
):
    root, _, _ = prepared_suite(tmp_path, monkeypatch)
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, command)
        return "101;ascend\n"

    monkeypatch.setattr(quality.subprocess, "check_output", sbatch)
    with pytest.raises(subprocess.CalledProcessError):
        quality.submit(root)
    suite = json.loads((root / "suite.json").read_text())
    assert suite["status"] == "submission_incomplete"
    assert suite["gates"][0]["job_id"] == "101"
    with pytest.raises(ValueError, match="partially submitted"):
        quality.submit(root)
    assert len(calls) == 2
