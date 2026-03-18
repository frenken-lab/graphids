"""Shared fixtures for training smoke and e2e tests."""

from __future__ import annotations

import os

import pytest
import torch
from torch_geometric.data import Data


def pytest_addoption(parser):
    parser.addoption(
        "--run-slurm",
        action="store_true",
        default=False,
        help="Run tests marked as needing SLURM compute node",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slurm: test requires SLURM compute node (CPU or GPU)")


def pytest_collection_modifyitems(config, items):
    on_compute = "SLURM_JOB_ID" in os.environ
    run_slurm = config.getoption("--run-slurm", default=False)
    if on_compute or run_slurm:
        return
    skip_slurm = pytest.mark.skip(
        reason="Needs SLURM compute node (use --run-slurm or submit via sbatch)"
    )
    for item in items:
        if "slurm" in item.keywords:
            item.add_marker(skip_slurm)


NUM_IDS = 20
IN_CHANNELS = 26


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_graph(num_nodes=10, num_edges=20, label=0):
    """Create a single synthetic graph matching real data shape."""
    from graphids.config import EDGE_FEATURE_COUNT

    x = torch.randn(num_nodes, IN_CHANNELS)
    x[:, 0] = torch.randint(0, NUM_IDS, (num_nodes,)).float()
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, EDGE_FEATURE_COUNT)
    y = torch.tensor([label])
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _make_dataset(n=50):
    """Create a small dataset with mix of normal/attack graphs."""
    return [_make_graph(label=i % 2) for i in range(n)]


# ---------------------------------------------------------------------------
# Shared smoke-test config overrides (nested format for new PipelineConfig)
# ---------------------------------------------------------------------------

SMOKE_OVERRIDES = dict(
    training=dict(
        max_epochs=2,
        batch_size=16,
        precision="32-true",
        patience=2,
        gradient_checkpointing=False,
        log_every_n_steps=1,
        safety_factor=1.0,
    ),
    device="cpu",
    num_workers=0,
    mp_start_method="spawn",
)

# E2E tests need tiny architectures to finish in reasonable time on CPU
E2E_OVERRIDES = dict(
    training=dict(
        max_epochs=2,
        batch_size=16,
        precision="32-true",
        patience=2,
        gradient_checkpointing=False,
        log_every_n_steps=1,
        safety_factor=1.0,
    ),
    device="cpu",
    num_workers=0,
    mp_start_method="spawn",
    # Tiny VGAE
    vgae=dict(hidden_dims=(32, 16, 8), latent_dim=8, heads=1, embedding_dim=4, dropout=0.1),
    # Tiny GAT
    gat=dict(hidden=8, layers=2, heads=2, embedding_dim=4, fc_layers=2, dropout=0.1),
    # Tiny DQN
    dqn=dict(hidden=32, layers=2, buffer_size=500, batch_size=32, target_update=10),
)
