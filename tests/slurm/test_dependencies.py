"""Tests for ``graphids/slurm/dependencies.py`` — ``--depends-on`` resolution.

The resolver maps a teacher variant (e.g. ``vgae``) to a downstream-preset
TLA name (e.g. ``vgae_ckpt_path``) via the :data:`DEPENDS_ON_TLA` registry,
and looks up state through MLflow tags. Dispatches FINISHED → ckpt-only
TLA, RUNNING → ckpt TLA + afterok jid (single primitive — see
``.claude/rules/single-submission-primitive.md``).
"""

from __future__ import annotations

import pandas as pd
import pytest
import typer

from graphids import _mlflow
from graphids.slurm.dependencies import (
    DEPENDS_ON_TLA,
    DependencyResolutionError,
    DependencySpec,
    parse_depends_on,
    resolve_all,
    resolve_dependency,
)

# ===========================================================================
# parse_depends_on
# ===========================================================================


def test_parse_depends_on_seed_fallback():
    specs = parse_depends_on("vgae,focal", default_seed=42)
    assert specs == [DependencySpec("vgae", 42), DependencySpec("focal", 42)]


def test_parse_depends_on_explicit_seed_overrides_default():
    assert parse_depends_on("vgae:99", default_seed=42) == [DependencySpec("vgae", 99)]


def test_parse_depends_on_non_integer_seed_raises():
    with pytest.raises(typer.BadParameter):
        parse_depends_on("vgae:abc", default_seed=None)


def test_parse_depends_on_missing_seed_raises():
    with pytest.raises(typer.BadParameter):
        parse_depends_on("vgae", default_seed=None)


# ===========================================================================
# resolve_dependency (MLflow mocked)
# ===========================================================================


def _patch_mlflow(monkeypatch, df: pd.DataFrame) -> None:
    monkeypatch.setattr(_mlflow, "ensure_tracking_uri", lambda: "sqlite:///x.db")
    monkeypatch.setattr(_mlflow.mlflow, "set_tracking_uri", lambda _u: None)
    monkeypatch.setattr(_mlflow.mlflow, "search_runs", lambda **_kw: df)


# CONTRACT: FINISHED upstream → (ckpt_path, None). No afterok jid needed
# because the ckpt already exists; downstream just reads it.
def test_resolve_dependency_finished_returns_ckpt_only(monkeypatch, tmp_path):
    run_dir = tmp_path / "vgae_run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints/best_model.ckpt").write_bytes(b"fake")

    df = pd.DataFrame(
        [{"run_id": "abc", "status": "FINISHED", "tags.graphids.run_dir": str(run_dir)}]
    )
    _patch_mlflow(monkeypatch, df)

    ckpt, jid = resolve_dependency(DependencySpec("vgae", 42), dataset="ds")
    assert ckpt == run_dir / "checkpoints/best_model.ckpt"
    assert jid is None


# CONTRACT: RUNNING upstream → (ckpt_path, jid). The ckpt may not exist on
# disk yet; downstream queues with afterok and the SLURM scheduler handles
# the wait. ckpt path comes from the run_dir tag (deterministic at fit-start).
def test_resolve_dependency_running_returns_ckpt_and_afterok(monkeypatch, tmp_path):
    run_dir = tmp_path / "vgae_running"
    # Intentionally do NOT create the checkpoint file — it doesn't exist yet
    # for a still-running upstream, but resolve_dependency must succeed.
    df = pd.DataFrame(
        [
            {
                "run_id": "abc",
                "status": "RUNNING",
                "tags.graphids.run_dir": str(run_dir),
                "tags.slurm.slurm_job_id": "123456",
            }
        ]
    )
    _patch_mlflow(monkeypatch, df)

    ckpt, jid = resolve_dependency(DependencySpec("vgae", 42), dataset="ds")
    assert ckpt == run_dir / "checkpoints/best_model.ckpt"
    assert jid == 123456


# CONTRACT: RUNNING upstream without slurm_job_id tag → typed error. Refuse
# to silently drop the dep; the user needs to know SLURM coupling is broken.
def test_resolve_dependency_running_without_jid_tag_raises(monkeypatch, tmp_path):
    run_dir = tmp_path / "vgae_running"
    df = pd.DataFrame(
        [{"run_id": "abc", "status": "RUNNING", "tags.graphids.run_dir": str(run_dir)}]
    )
    _patch_mlflow(monkeypatch, df)
    with pytest.raises(DependencyResolutionError, match="slurm_job_id"):
        resolve_dependency(DependencySpec("vgae", 42), dataset="ds")


# CONTRACT: terminal-bad statuses (FAILED, KILLED) raise. Re-submit is required.
@pytest.mark.parametrize("status", ["FAILED", "KILLED", "SCHEDULED"])
def test_resolve_dependency_bad_status_raises(monkeypatch, status):
    df = pd.DataFrame([{"run_id": "abc", "status": status, "tags.graphids.run_dir": "/x"}])
    _patch_mlflow(monkeypatch, df)
    with pytest.raises(DependencyResolutionError, match=f"status is {status!r}"):
        resolve_dependency(DependencySpec("vgae", 42), dataset="ds")


def test_resolve_dependency_no_match_raises(monkeypatch):
    _patch_mlflow(monkeypatch, pd.DataFrame([]))
    with pytest.raises(DependencyResolutionError, match="Submit it first"):
        resolve_dependency(DependencySpec("vgae", 42), dataset="ds")


# REGRESSION: pre-2026-04 runs lack ``graphids.run_dir`` — must surface a typed
# error with a recovery hint, not leak ``TypeError`` from ``Path(None)``.
def test_resolve_dependency_missing_run_dir_tag_raises(monkeypatch):
    df = pd.DataFrame([{"run_id": "abc", "status": "FINISHED"}])
    _patch_mlflow(monkeypatch, df)
    with pytest.raises(DependencyResolutionError, match="no graphids.run_dir tag"):
        resolve_dependency(DependencySpec("vgae", 42), dataset="ds")


# CONTRACT: FINISHED in MLflow but ckpt missing on disk → typed error.
def test_resolve_dependency_finished_ckpt_missing_on_disk_raises(monkeypatch, tmp_path):
    run_dir = tmp_path / "vgae_run"
    run_dir.mkdir()
    df = pd.DataFrame(
        [{"run_id": "abc", "status": "FINISHED", "tags.graphids.run_dir": str(run_dir)}]
    )
    _patch_mlflow(monkeypatch, df)
    with pytest.raises(DependencyResolutionError, match="ckpt missing on disk"):
        resolve_dependency(DependencySpec("vgae", 42), dataset="ds")


# ===========================================================================
# resolve_all
# ===========================================================================


# CONTRACT: vgae → vgae_ckpt_path, focal → gat_ckpt_path. Catches a
# silent registry swap that would mis-wire every fusion run's teachers.
def test_resolve_all_uses_correct_registry_tla_names(monkeypatch, tmp_path):
    def _mk(name: str) -> str:
        d = tmp_path / name
        (d / "checkpoints").mkdir(parents=True)
        (d / "checkpoints/best_model.ckpt").write_bytes(b"x")
        return str(d)

    vgae_dir = _mk("vgae")
    focal_dir = _mk("focal")

    def fake_search(**kw):
        f = kw["filter_string"]
        if "graphids.variant` = 'vgae'" in f:
            return pd.DataFrame(
                [{"run_id": "v", "status": "FINISHED", "tags.graphids.run_dir": vgae_dir}]
            )
        if "graphids.variant` = 'focal'" in f:
            return pd.DataFrame(
                [{"run_id": "f", "status": "FINISHED", "tags.graphids.run_dir": focal_dir}]
            )
        return pd.DataFrame([])

    monkeypatch.setattr(_mlflow, "ensure_tracking_uri", lambda: "sqlite:///x.db")
    monkeypatch.setattr(_mlflow.mlflow, "set_tracking_uri", lambda _u: None)
    monkeypatch.setattr(_mlflow.mlflow, "search_runs", fake_search)

    tlas, jids = resolve_all(
        [DependencySpec("vgae", 42), DependencySpec("focal", 42)], dataset="ds"
    )
    by_name = dict(tlas)
    assert by_name["vgae_ckpt_path"] == f"{vgae_dir}/checkpoints/best_model.ckpt"
    assert by_name["gat_ckpt_path"] == f"{focal_dir}/checkpoints/best_model.ckpt"
    assert jids == []  # both FINISHED → no afterok needed


# CONTRACT: a mix of FINISHED + RUNNING upstreams populates BOTH return lists.
# REGRESSION risk: dropping the jid would silently submit downstream too early
# if upstream hadn't actually written the ckpt yet.
def test_resolve_all_mixed_finished_and_running(monkeypatch, tmp_path):
    vgae_dir = tmp_path / "vgae"
    (vgae_dir / "checkpoints").mkdir(parents=True)
    (vgae_dir / "checkpoints/best_model.ckpt").write_bytes(b"x")
    focal_dir = tmp_path / "focal_running"

    def fake_search(**kw):
        f = kw["filter_string"]
        if "graphids.variant` = 'vgae'" in f:
            return pd.DataFrame(
                [{"run_id": "v", "status": "FINISHED", "tags.graphids.run_dir": str(vgae_dir)}]
            )
        if "graphids.variant` = 'focal'" in f:
            return pd.DataFrame(
                [
                    {
                        "run_id": "f",
                        "status": "RUNNING",
                        "tags.graphids.run_dir": str(focal_dir),
                        "tags.slurm.slurm_job_id": "987654",
                    }
                ]
            )
        return pd.DataFrame([])

    monkeypatch.setattr(_mlflow, "ensure_tracking_uri", lambda: "sqlite:///x.db")
    monkeypatch.setattr(_mlflow.mlflow, "set_tracking_uri", lambda _u: None)
    monkeypatch.setattr(_mlflow.mlflow, "search_runs", fake_search)

    tlas, jids = resolve_all(
        [DependencySpec("vgae", 42), DependencySpec("focal", 42)], dataset="ds"
    )
    assert dict(tlas) == {
        "vgae_ckpt_path": f"{vgae_dir}/checkpoints/best_model.ckpt",
        "gat_ckpt_path": f"{focal_dir}/checkpoints/best_model.ckpt",
    }
    assert jids == [987654]


def test_resolve_all_unknown_variant_raises():
    with pytest.raises(typer.BadParameter, match="not in dependency registry"):
        resolve_all([DependencySpec("zzzz_unknown", 42)], dataset="ds")


# ===========================================================================
# Registry / plan drift catcher
# ===========================================================================


# INVARIANT: every entry in DEPENDS_ON_TLA points at a real fit-node variant
# in the shipped OFAT plan. Catches "renamed a variant + forgot to update the
# registry". The plan jsonnet IS the topology source of truth.
def test_registry_entries_exist_in_ofat_plan():
    from graphids.config.jsonnet import render
    from graphids.slurm.dag import parse_plan

    nodes = parse_plan(render("configs/plans/ofat.jsonnet", tla={"dataset": "x", "seed": 0}))
    fit_variants = {n.variant for n in nodes if n.preset_path and n.action == "fit"}
    orphaned = set(DEPENDS_ON_TLA) - fit_variants
    assert not orphaned, f"DEPENDS_ON_TLA points at non-existent variants: {orphaned}"
