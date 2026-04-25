"""Tests for the status-gated resume matrix in ``_mlflow.start_training_run``.

The decision logic (`_resume_decision`) is pure; these tests verify the
full search-then-start path against a real SQLite-backed MLflow store in
``tmp_path``. Five status transitions + git-SHA discontinuity + the
``GRAPHIDS_FORCE_RESUME`` escape hatch.

Why test this: the resume logic races with ``mlflow_reap_zombies`` at the
``TERMINATED`` seam and with live training processes at ``RUNNING`` —
wrong decisions silently corrupt history (see moderate plan Q5).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from graphids._mlflow import _resume_decision


def _mk(status: str, sha: str | None = None):
    """Lightweight Run stand-in — matches the two fields _resume_decision reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        info=SimpleNamespace(status=status),
        data=SimpleNamespace(tags={"git_sha": sha} if sha else {}),
    )


# ----- Decision matrix (unit) ------------------------------------------------


# REGRESSION: zombie TERMINATED rows were being un-TERMINATED by a naive
# "always resume" policy, bypassing the reap audit trail.
@pytest.mark.parametrize(
    "status, force, expected",
    [
        ("FAILED", False, "resume"),
        ("KILLED", False, "resume"),
        ("TERMINATED", False, "new"),  # reaper owns tombstone
        ("RUNNING", False, "refuse"),  # live writer or pre-reaper zombie
        ("FINISHED", False, "refuse"),  # done; need --force-resume
    ],
)
def test_resume_decision_no_force(status: str, force: bool, expected: str) -> None:
    assert _resume_decision(_mk(status), None, force) == expected


# CONTRACT: --force-resume escape hatch lets a human override TERMINATED /
# FINISHED for retests or recovery. RUNNING still refuses — never race a
# live writer regardless of force.
@pytest.mark.parametrize(
    "status, expected",
    [
        ("FAILED", "resume"),
        ("KILLED", "resume"),
        ("TERMINATED", "resume"),
        ("RUNNING", "refuse"),
        ("FINISHED", "resume"),
    ],
)
def test_resume_decision_force(status: str, expected: str) -> None:
    assert _resume_decision(_mk(status), None, True) == expected


# INVARIANT (review Q6, option b): git-SHA change forces a new run rather
# than silently mixing commits in one row. Force lets cross-SHA resume
# through (researcher took responsibility).
def test_git_sha_change_forces_new_run() -> None:
    assert _resume_decision(_mk("FAILED", "aaa"), "bbb", False) == "new"
    assert _resume_decision(_mk("FAILED", "aaa"), "aaa", False) == "resume"
    assert _resume_decision(_mk("FAILED", "aaa"), "bbb", True) == "resume"


# ----- End-to-end against real SQLite MLflow ---------------------------------


@pytest.fixture
def mlflow_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point graphids at a tmp SQLite MLflow backend; null out cluster env.

    Cluster env (``GRAPHIDS_CLUSTER`` / ``SLURM_CLUSTER_NAME``) is nulled so
    ``run_name_for(identity, cluster=None)`` matches what the tests seed.
    Under SLURM test runs, ``SLURM_CLUSTER_NAME`` would otherwise make the
    run_name include a ``_pitzer`` suffix and resume lookup would miss.
    Also drops the settings cache so the next ``get_settings()`` re-reads
    the patched env.
    """
    db = tmp_path / "mlflow.db"
    uri = f"sqlite:///{db}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("GRAPHIDS_FORCE_RESUME", raising=False)
    monkeypatch.delenv("GRAPHIDS_CLUSTER", raising=False)
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    from graphids.config.settings import get_settings

    get_settings.cache_clear()

    import mlflow

    mlflow.set_tracking_uri(uri)
    return tmp_path


def _seed_run(
    experiment: str,
    run_name: str,
    status: str,
    *,
    git_sha: str | None = None,
    phase: str = "fit",
) -> str:
    """Create a run in the given status and return its run_id."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        exp_id = client.create_experiment(experiment)
    except Exception:
        exp_id = client.get_experiment_by_name(experiment).experiment_id
    run = client.create_run(exp_id, run_name=run_name, tags={"graphids.phase": phase})
    if git_sha:
        client.set_tag(run.info.run_id, "git_sha", git_sha)
    if status != "RUNNING":
        client.set_terminated(run.info.run_id, status=status)
    return run.info.run_id


# REGRESSION: a FAILED run's run_id must be reused by the next attempt so
# the variant ends with ONE MLflow row, not three.
def test_resume_failed_reuses_run_id(mlflow_tmp: Path) -> None:
    from graphids._mlflow import start_training_run

    run_dir = mlflow_tmp / "set_01" / "ablations" / "unsupervised" / "vgae" / "seed_42"
    run_dir.mkdir(parents=True)
    experiment = "graphids/set_01/unsupervised"
    run_name = "unsupervised_vgae_set_01_seed42"

    seeded = _seed_run(experiment, run_name, status="FAILED")
    result = start_training_run(run_dir, resolved_config={})
    assert result == run_name

    import mlflow

    active = mlflow.active_run()
    assert active is not None, "start_training_run must leave a run active"
    assert active.info.run_id == seeded, (
        f"expected resume of run_id={seeded}, got {active.info.run_id}"
    )
    mlflow.end_run()


# REGRESSION: FINISHED must not silently resume — that clobbers an
# already-good row. GRAPHIDS_FORCE_RESUME=1 is the explicit opt-in.
def test_finished_refuses_without_force(mlflow_tmp: Path) -> None:
    from graphids._mlflow import start_training_run

    run_dir = mlflow_tmp / "set_01" / "ablations" / "unsupervised" / "vgae" / "seed_42"
    run_dir.mkdir(parents=True)
    _seed_run("graphids/set_01/unsupervised", "unsupervised_vgae_set_01_seed42", "FINISHED")

    assert start_training_run(run_dir, resolved_config={}) is None

    import mlflow

    assert mlflow.active_run() is None, "refusal must not leave an active run behind"


def test_finished_resumes_with_force(mlflow_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from graphids._mlflow import start_training_run

    run_dir = mlflow_tmp / "set_01" / "ablations" / "unsupervised" / "vgae" / "seed_42"
    run_dir.mkdir(parents=True)
    seeded = _seed_run(
        "graphids/set_01/unsupervised", "unsupervised_vgae_set_01_seed42", "FINISHED"
    )
    monkeypatch.setenv("GRAPHIDS_FORCE_RESUME", "1")

    result = start_training_run(run_dir, resolved_config={})
    assert result == "unsupervised_vgae_set_01_seed42"

    import mlflow
    from mlflow.tracking import MlflowClient

    active = mlflow.active_run()
    assert active is not None and active.info.run_id == seeded
    # ``active.data.tags`` is a snapshot at ``start_run`` time and doesn't
    # pick up subsequent ``set_tags`` writes; refetch via the client.
    fresh = MlflowClient().get_run(seeded)
    assert fresh.data.tags.get("graphids.resume.forced") == "true"
    mlflow.end_run()


# REGRESSION: RUNNING must refuse regardless of force — protects against
# racing a live training process on the same row.
def test_running_refuses_even_with_force(mlflow_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from graphids._mlflow import start_training_run

    run_dir = mlflow_tmp / "set_01" / "ablations" / "unsupervised" / "vgae" / "seed_42"
    run_dir.mkdir(parents=True)
    _seed_run("graphids/set_01/unsupervised", "unsupervised_vgae_set_01_seed42", "RUNNING")
    monkeypatch.setenv("GRAPHIDS_FORCE_RESUME", "1")

    assert start_training_run(run_dir, resolved_config={}) is None


# ----- Upstream lineage (pure) -----------------------------------------------


# INVARIANT (review Q3 + #3): upstream tags key on filesystem ``run_dir``
# not MLflow ``run_id`` — stable across tracking-tool swaps, no query needed
# at submit time. Role derived from the standard ablation tree.
def test_upstream_tags_curriculum_vgae_shape() -> None:
    from graphids._mlflow import _upstream_tags

    cfg = {
        "data": {
            "init_args": {
                "scorer": {
                    "init_args": {
                        "ckpt_path": "/lake/set_01/ablations/unsupervised/vgae/seed_42/checkpoints/best_model.ckpt"
                    }
                }
            }
        }
    }
    tags = _upstream_tags(cfg)
    assert tags["graphids.upstream.unsupervised_vgae.run_dir"] == (
        "/lake/set_01/ablations/unsupervised/vgae/seed_42"
    )
    assert tags["graphids.upstream.unsupervised_vgae.ckpt_path"].endswith("best_model.ckpt")


def test_upstream_tags_empty_when_no_ckpt() -> None:
    from graphids._mlflow import _upstream_tags

    assert _upstream_tags({}) == {}
    assert _upstream_tags({"data": {"init_args": {"something": 42}}}) == {}


# ----- LoggedModel registration (e2e) ----------------------------------------


# CONTRACT (maximalist Feature 4): fit-end registers a LoggedModel with
# source_run_id + model_type + ckpt tags. Downstream fusion queries via
# search_logged_models(source_run_id=..., model_type=...) → model_id.
def test_register_logged_model_writes_entity(mlflow_tmp: Path) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    from graphids._mlflow import RunIdentity, _register_logged_model

    experiment = "graphids/set_01/unsupervised"
    run_name = "unsupervised_vgae_set_01_seed42"
    run_id = _seed_run(experiment, run_name, "FAILED")

    run_dir = mlflow_tmp / "set_01" / "ablations" / "unsupervised" / "vgae" / "seed_42"
    run_dir.mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / "best_model.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"not-a-real-ckpt")

    mlflow.set_experiment(experiment)
    exp = MlflowClient().get_experiment_by_name(experiment)
    ident = RunIdentity(group="unsupervised", variant="vgae", dataset="set_01", seed=42)
    model_id = _register_logged_model(run_id, exp.experiment_id, ident, run_dir, str(ckpt))
    assert model_id is not None

    hits = MlflowClient().search_logged_models(
        experiment_ids=[exp.experiment_id],
        filter_string=f"source_run_id = '{run_id}' AND model_type = 'unsupervised_vgae'",
        max_results=1,
    )
    assert len(hits) == 1 and hits[0].model_id == model_id
    assert hits[0].tags.get("graphids.ckpt_path") == str(ckpt)
    assert hits[0].tags.get("graphids.run_dir") == str(run_dir)
