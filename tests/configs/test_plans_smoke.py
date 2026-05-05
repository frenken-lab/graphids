"""Smoke test — every Python plan builds a valid Blueprint.

Replaces the differential parity tests after the jsonnet sources were
deleted (Step 6 of the migration). Verifies each plan's ``build()``
returns a list that passes ``BlueprintArray`` validation — catches
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
    """``build(dataset, seed)`` returns rows that pass BlueprintArray validation."""
    from graphids.plan.blueprint import BlueprintArray

    mod = importlib.import_module(f"graphids.plan.plans.{plan_name}")
    rows = mod.build(dataset="hcrl_sa", seed=42)
    assert isinstance(rows, list) and len(rows) > 0
    blueprint = BlueprintArray.model_validate(rows)
    assert len(blueprint) == len(rows)
