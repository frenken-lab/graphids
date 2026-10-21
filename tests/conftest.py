"""Shared test execution settings."""

from __future__ import annotations

import os

import pytest

# Synthetic CPU cases are slower when PyTorch starts a large thread pool.
# Set limits before test modules import torch; NumPy-only tests need no torch.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        return
    skip = pytest.mark.skip(reason="requires SLURM allocation (SLURM_JOB_ID unset)")
    for item in items:
        if "slurm" in item.keywords:
            item.add_marker(skip)
