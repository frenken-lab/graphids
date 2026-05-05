"""Smoke test — every Python plan builds a valid Blueprint.

Replaces the differential parity tests after the jsonnet sources were
deleted (Step 6 of the migration). Verifies each plan's ``build()``
returns a list that passes ``Plan`` validation — catches
schema drift, broken imports, and missing primitives at test time
rather than at submit time.

Login-node safe: pure render, no torch.
"""

from __future__ import annotations

import importlib
import os

import pytest

# Force registration of all CLI commands on the root Typer `app`. Decorators
# in commands.py / plans.py / shortcuts.py only fire on import; without these
# the test runner sees an empty app.
import graphids.cli.commands  # noqa: F401
import graphids.cli.plans  # noqa: F401

PLANS = [
    "ablations.unsupervised",
    "ablations.fusion",
    "ablations.supervised",
    "smoke.gat_taunorm",
    "data.rebuild_cache",
]


@pytest.fixture(scope="module")
def env_roots(tmp_path_factory):
    saved = {k: os.environ.get(k) for k in ("GRAPHIDS_RUN_ROOT", "GRAPHIDS_LAKE_ROOT")}
    os.environ["GRAPHIDS_RUN_ROOT"] = str(tmp_path_factory.mktemp("run_root"))
    os.environ["GRAPHIDS_LAKE_ROOT"] = str(tmp_path_factory.mktemp("lake_root"))
    from graphids.paths import load_catalog

    load_catalog.cache_clear()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.parametrize("plan_name", PLANS)
def test_plan_builds(plan_name: str, env_roots):
    """``build(dataset, seed)`` returns rows that pass ``Plan`` validation."""
    from graphids.plan.identity import mint_plan_id
    from graphids.plan.schema import Plan

    mod = importlib.import_module(f"graphids.plan.plans.{plan_name}")
    rows = mod.build(dataset="hcrl_sa", seed=42)
    assert isinstance(rows, list) and len(rows) > 0
    plan_id = mint_plan_id()
    for r in rows:
        r["plan_id"] = plan_id
        if r.get("action") in {"fit", "test"}:
            r["plan_module"] = plan_name
            r["git_sha"] = "test_sha"
    plan_obj = Plan.model_validate({
        "plan_id": plan_id,
        "plan_module": plan_name,
        "plan_args": {"dataset": "hcrl_sa", "seed": 42},
        "created_at": "2026-05-05T00:00:00+00:00",
        "rows": rows,
    })
    assert len(plan_obj) == len(rows)


def test_run_filter_subsets_rows(env_roots):
    """`graphids run --filter <glob>` renders only matching rows.

    CONTRACT: replaces the killed `plans retry` via composition. A
    single-row retry is `--filter <exact-name>` + iteration.
    """
    from typer.testing import CliRunner

    from graphids.cli.app import app

    runner = CliRunner()
    full = runner.invoke(
        app, ["run", "smoke.gat_taunorm", "--dataset", "hcrl_sa", "--seed", "42"]
    )
    assert full.exit_code == 0, full.stderr
    full_names = [r["name"] for r in __import__("json").loads(full.stdout)["rows"]]
    assert len(full_names) >= 2

    target = full_names[0]
    filtered = runner.invoke(
        app,
        ["run", "smoke.gat_taunorm", "--dataset", "hcrl_sa", "--seed", "42",
         "--filter", target],
    )
    assert filtered.exit_code == 0, filtered.stderr
    filtered_rows = __import__("json").loads(filtered.stdout)["rows"]
    assert [r["name"] for r in filtered_rows] == [target]


def test_identity_tags_carry_reproduction_contract(env_roots):
    """`identity_tags(row, phase)` emits the five reproduction-contract tags.

    CONTRACT: `git checkout <git_sha> && graphids run <plan_module> --dataset <dataset>
    --seed <seed> --filter <row_name>` regenerates this exact row. The five tags
    are the inputs to that command. Missing one breaks reproduction silently.
    """
    from graphids._mlflow import identity_tags
    from graphids.plan.identity import mint_plan_id
    from graphids.plan.schema import Plan

    mod = importlib.import_module("graphids.plan.plans.smoke.gat_taunorm")
    rows = mod.build(dataset="hcrl_sa", seed=42)
    plan_id = mint_plan_id()
    for r in rows:
        r["plan_id"] = plan_id
        if r.get("action") in {"fit", "test"}:
            r["plan_module"] = "smoke.gat_taunorm"
            r["git_sha"] = "abc123def456"
    plan_obj = Plan.model_validate({
        "plan_id": plan_id,
        "plan_module": "smoke.gat_taunorm",
        "plan_args": {"dataset": "hcrl_sa", "seed": 42},
        "created_at": "2026-05-05T00:00:00+00:00",
        "rows": rows,
    })
    fit_row = next(r for r in plan_obj.rows if r.action == "fit")
    tags = identity_tags(fit_row, "fit")

    assert tags["graphids.plan_id"] == plan_id
    assert tags["graphids.plan_module"] == "smoke.gat_taunorm"
    assert tags["graphids.git_sha"] == "abc123def456"
    assert tags["graphids.row_name"] == fit_row.name
    assert tags["graphids.dataset"] == "hcrl_sa"
    assert tags["graphids.seed"] == "42"


def test_run_filter_no_match_errors_with_available_names(env_roots):
    """No-match raises with the list of available row names."""
    from typer.testing import CliRunner

    from graphids.cli.app import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", "smoke.gat_taunorm", "--dataset", "hcrl_sa", "--seed", "42",
         "--filter", "nonexistent_row_*"],
    )
    assert result.exit_code != 0
    assert "matched 0/" in result.stderr
    assert "Available:" in result.stderr


def _render_plan_to(env_roots, tmp_path) -> object:
    """Helper: render `smoke.gat_taunorm` to a tempdir, return Path."""
    import json as _json
    from typer.testing import CliRunner

    from graphids.cli.app import app

    runner = CliRunner()
    plan_path = tmp_path / "plan.json"
    res = runner.invoke(
        app,
        ["run", "smoke.gat_taunorm", "--dataset", "hcrl_sa", "--seed", "42",
         "-o", str(plan_path)],
    )
    assert res.exit_code == 0, res.stderr
    assert plan_path.exists()
    # sanity — the rows have plan_module + git_sha threaded
    plan = _json.loads(plan_path.read_text())
    for row in plan["rows"]:
        if row["action"] in {"fit", "test"}:
            assert "plan_module" in row and "git_sha" in row
    return plan_path


def test_plans_submit_dry_run_lists_all_rows(env_roots, tmp_path):
    """`plans submit --dry-run` enumerates every row without submitting."""
    from typer.testing import CliRunner
    from graphids.cli.app import app

    plan_path = _render_plan_to(env_roots, tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["plans", "submit", "--plan", str(plan_path), "-C", "pitzer", "--dry-run"],
    )
    assert result.exit_code == 0, result.stderr
    assert "would-submit=2" in result.stdout
    assert "gat_taunorm" in result.stdout
    assert "gat_taunorm-test" in result.stdout


def test_plans_submit_filter_subsets_rows_for_dry_run(env_roots, tmp_path):
    """`plans submit --filter <glob> --dry-run` only enumerates matching rows."""
    from typer.testing import CliRunner
    from graphids.cli.app import app

    plan_path = _render_plan_to(env_roots, tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["plans", "submit", "--plan", str(plan_path), "-C", "pitzer",
         "--filter", "*-test", "--dry-run"],
    )
    assert result.exit_code == 0, result.stderr
    assert "would-submit=1" in result.stdout
    assert "gat_taunorm-test" in result.stdout
    # the fit row was filtered out
    fit_lines = [
        line for line in result.stdout.splitlines()
        if "would-submit" in line and "gat_taunorm " in line and "-test" not in line
    ]
    assert fit_lines == [], f"fit row leaked through filter: {fit_lines}"


def test_plans_submit_filter_no_match_errors(env_roots, tmp_path):
    """`plans submit --filter <glob>` with 0 matches errors with available names."""
    from typer.testing import CliRunner
    from graphids.cli.app import app

    plan_path = _render_plan_to(env_roots, tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["plans", "submit", "--plan", str(plan_path), "-C", "pitzer",
         "--filter", "no_such_row_*", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "matched 0/" in result.stderr
    assert "Available:" in result.stderr
