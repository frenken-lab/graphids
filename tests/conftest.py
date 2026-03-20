"""Shared fixtures for orchestration tests.

Expensive objects (DAG topology, Dagster assets) are built once per module.
Resource spec factories avoid repeating timedelta boilerplate.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from graphids.pipeline.orchestration.job import ResourceSpec


# ---------------------------------------------------------------------------
# Resource spec factories
# ---------------------------------------------------------------------------


@pytest.fixture()
def gpu_resources() -> ResourceSpec:
    """Standard GPU resource spec for tests."""
    return ResourceSpec(
        partition="gpu", gpus=1, cpus=4,
        memory_gb=16, walltime=timedelta(hours=3),
    )


@pytest.fixture()
def cpu_resources() -> ResourceSpec:
    """Standard CPU resource spec for tests."""
    return ResourceSpec(
        partition="cpu", gpus=0, cpus=8,
        memory_gb=32, walltime=timedelta(hours=1),
    )


# ---------------------------------------------------------------------------
# DAG topology (built once per module that requests it)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dag_topology():
    """Pipeline DAG topology — built once, shared across all tests in a module."""
    from graphids.pipeline.orchestration.dagster_defs import build_dag_topology

    return build_dag_topology()


@pytest.fixture(scope="module")
def dagster_assets():
    """Dagster asset definitions — built once, shared across all tests in a module."""
    from graphids.pipeline.orchestration.dagster_defs import build_dagster_assets

    return build_dagster_assets()


# ---------------------------------------------------------------------------
# Asset lookup helpers (exposed as fixtures for cleaner test signatures)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def find_asset(dagster_assets):
    """Return a lookup function: find_asset(name) -> asset definition."""
    index = {a.key.path[-1]: a for a in dagster_assets}

    def _find(name: str):
        return index[name]

    return _find


@pytest.fixture(scope="module")
def asset_dep_names(dagster_assets):
    """Return a lookup function: asset_dep_names(name) -> set of dep names."""
    index = {a.key.path[-1]: a for a in dagster_assets}

    def _deps(name: str) -> set[str]:
        spec = list(index[name].specs)[0]
        return {d.asset_key.path[-1] for d in spec.deps}

    return _deps
