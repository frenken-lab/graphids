"""Tests for ``graphids._mlflow.resume_state`` — status-gated resume policy.

Five branches: no experiment, FINISHED, FAILED, RUNNING, RUNNING+FORCE.
Verifies against a real SQLite-backed MLflow store in ``tmp_path`` —
no client mocks; the search/decision path runs end-to-end.

Why test this: wrong decisions silently corrupt history. ``RUNNING`` →
``new`` would race a live job; ``FAILED`` → ``new`` would lose resume
state. Each branch is one assertion; no formula mirroring.
"""

from __future__ import annotations

import pytest
from mlflow.tracking import MlflowClient

from graphids._mlflow import end_training_run, resume_state, start_training_run
from graphids.blueprint import Identity, Meta, Resources, TrainRow


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh sqlite tracking server per test, with `_TRACKING_SET` reset."""
    import graphids._mlflow as m

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    m._TRACKING_SET = False
    return MlflowClient()


@pytest.fixture
def row():
    return TrainRow(
        name="focal",
        action="fit",
        identity=Identity(
            run_name="gat_loss_focal_hcrl_sa_seed42",
            run_dir="/r/hcrl_sa/ablations/gat_loss/focal/seed_42",
            jobname="gat-small-focal",
        ),
        meta=Meta(
            group="gat_loss",
            variant="focal",
            dataset="hcrl_sa",
            seed=42,
            model_type="gat",
            scale="small",
        ),
        rendered_config={},
        upstreams=[],
        resources=Resources(mode="gpu", length="long"),
    )


_KEYS = {"dataset": "hcrl_sa", "group": "gat_loss", "variant": "focal", "seed": 42}


# REGRESSION: missing experiment used to raise on the search_runs call.
def test_no_experiment_yet(client):
    d = resume_state(client, **_KEYS)
    assert d.action == "new" and d.run_id is None


# CONTRACT: FINISHED prior fit → fresh run for re-train. Resubmit-means-redo,
# so a stale FINISHED can't silently mask a new attempt.
def test_finished_yields_new(client, row):
    rid = start_training_run(row, phase="fit")
    end_training_run(rid, "FINISHED")
    d = resume_state(client, **_KEYS)
    assert d.action == "new" and d.run_id is None


# REGRESSION: FAILED → resume same run_id so MLflow history stays linear
# rather than splitting across N rows for the same (variant, seed).
def test_failed_yields_resume_same_run_id(client, row):
    rid = start_training_run(row, phase="fit")
    end_training_run(rid, "FAILED")
    d = resume_state(client, **_KEYS)
    assert d.action == "resume" and d.run_id == rid


# INVARIANT: RUNNING is opaque — could be live, could be zombie. Default
# is refuse; only GRAPHIDS_FORCE_RESUME=1 overrides.
def test_running_refuses(client, row, monkeypatch):
    monkeypatch.delenv("GRAPHIDS_FORCE_RESUME", raising=False)
    rid = start_training_run(row, phase="fit")
    d = resume_state(client, **_KEYS)
    assert d.action == "refuse" and d.run_id == rid


# CONTRACT: GRAPHIDS_FORCE_RESUME=1 escapes the refuse branch — for when
# the operator confirms a stale RUNNING (e.g. SLURM kill without FAILED
# transition) and wants to pick up from the prior run_id.
def test_running_with_force_resumes(client, row, monkeypatch):
    monkeypatch.setenv("GRAPHIDS_FORCE_RESUME", "1")
    rid = start_training_run(row, phase="fit")
    d = resume_state(client, **_KEYS)
    assert d.action == "resume" and d.run_id == rid


# CONTRACT: reason is a non-empty human string. Cheap invariant; logs and
# error messages always carry a why.
def test_decision_carries_reason(client, row):
    end_training_run(start_training_run(row, phase="fit"), "FAILED")
    assert resume_state(client, **_KEYS).reason
