"""Shared test execution settings."""

from __future__ import annotations

import os

import pytest
import torch

# Synthetic CPU cases are slower when PyTorch starts a large thread pool.
torch.set_num_threads(1)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        return
    skip = pytest.mark.skip(reason="requires SLURM allocation (SLURM_JOB_ID unset)")
    for item in items:
        if "slurm" in item.keywords:
            item.add_marker(skip)
