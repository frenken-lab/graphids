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

PLANS = [
    "ablations.unsupervised",
    "ablations.fusion",
    "ablations.ofat",
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
    from graphids.cli.commands import mint_plan_id
    from graphids.plan.blueprint import Plan

    mod = importlib.import_module(f"graphids.plan.plans.{plan_name}")
    rows = mod.build(dataset="hcrl_sa", seed=42)
    assert isinstance(rows, list) and len(rows) > 0
    plan_id = mint_plan_id()
    for r in rows:
        r["plan_id"] = plan_id
    plan_obj = Plan.model_validate({
        "plan_id": plan_id,
        "plan_module": plan_name,
        "plan_args": {"dataset": "hcrl_sa", "seed": 42},
        "created_at": "2026-05-05T00:00:00+00:00",
        "rows": rows,
    })
    assert len(plan_obj) == len(rows)
