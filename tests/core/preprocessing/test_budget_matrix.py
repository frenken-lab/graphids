"""Budget matrix tests: exercise node_budget() across real project configurations.

Parametrized over datasets, model types/scales, GPU types, and worker counts
using realistic probe values estimated from model architectures. These are
CPU tests (mock _probe) — actual GPU validation is a separate SLURM job.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from graphids.core.preprocessing.budget import (
    BudgetResult,
    _SAFETY_MARGIN,
    node_budget,
)


# --- Real dataset statistics (from cache_metadata.json) ----------------------

DATASETS = {
    # name: (mean_nodes, p95_nodes)
    "hcrl_ch": (21.9, 24),
    "hcrl_sa": (36.5, 63),
    "set_01":  (28.2, 35),
    "set_02":  (34.2, 40),
}

# --- Realistic probe values per model type + scale --------------------------
# Estimated from architecture params. γ (collation) is model-independent
# (same PyG Batch.from_data_list). β and bytes_per_node scale with model size.
# α is GPU overhead per step (kernel launch, scheduler).
# These are ESTIMATES — real values come from running the probe on GPU.

MODEL_PROBES = {
    # (model_type, scale): (bytes_per_node, gamma_s, alpha_s, beta_s, conv_type)
    #
    # VGAE small: 3 layers, hidden=[80,40,16], heads=1, ~24K params
    ("vgae", "small"):    (1200,  70e-6, 0.002, 0.10e-6, "gatv2"),
    # VGAE large: 3 layers, hidden=[480,240,64], heads=4, ~200K params
    ("vgae", "large"):    (4500,  70e-6, 0.005, 0.80e-6, "gatv2"),
    # GAT small: 2 layers, hidden=24, heads=4, ~15K params
    ("gat", "small"):     (800,   70e-6, 0.002, 0.08e-6, "gatv2"),
    # GAT large: 3 layers, hidden=64, heads=4, ~60K params
    ("gat", "large"):     (3200,  70e-6, 0.004, 0.50e-6, "gatv2"),
    # DGI small: same arch as VGAE small + discriminator
    ("dgi", "small"):     (1500,  70e-6, 0.003, 0.12e-6, "gatv2"),
    # DGI large: same arch as VGAE large + discriminator
    ("dgi", "large"):     (5000,  70e-6, 0.006, 0.90e-6, "gatv2"),
    # Temporal small: 2 spatial + 2 transformer layers
    ("temporal", "small"): (4000, 70e-6, 0.008, 1.50e-6, "gatv2"),
    # Temporal large: 3 spatial + 3 transformer layers
    ("temporal", "large"): (9000, 70e-6, 0.015, 4.00e-6, "gatv2"),
}

# --- GPU free VRAM after model load -----------------------------------------

GPU_TYPES = {
    # name: free_bytes (after model + CUDA context)
    "v100_16gb":  14 * 1024**3,   # Pitzer
    "a100_40gb":  36 * 1024**3,   # Ascend
    "a100_80gb":  76 * 1024**3,   # Ascend
}

WORKER_COUNTS = [2, 6, 8]


# --- Helpers -----------------------------------------------------------------

def _run(tmp_path, *, dataset, model_key, gpu, num_workers):
    """Run node_budget with mocked probe and VRAM."""
    mean_nodes, _ = DATASETS[dataset]
    bpn, gamma, alpha, beta, conv_type = MODEL_PROBES[model_key]
    free = GPU_TYPES[gpu]

    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text(f'{{"graph_stats":{{"node_count":{{"mean":{mean_nodes}}}}}}}')

    def mock_probe(model, ds, step_fn=None):
        return bpn, gamma, alpha, beta

    with (
        patch("graphids.core.preprocessing.budget.cache_dir", return_value=tmp_path),
        patch("graphids.core.preprocessing.budget._probe", mock_probe),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(free, free)),
    ):
        return node_budget(
            dataset, str(tmp_path), conv_type=conv_type,
            model=True, train_dataset=True, num_workers=num_workers,
        )


# --- Parametrized tests ------------------------------------------------------

_all_combos = [
    (ds, mk, gpu, w)
    for ds in DATASETS
    for mk in MODEL_PROBES
    for gpu in GPU_TYPES
    for w in WORKER_COUNTS
]

# Readable test IDs: "set_01-gat_small-v100_16gb-w6"
_ids = [f"{ds}-{mk[0]}_{mk[1]}-{gpu}-w{w}" for ds, mk, gpu, w in _all_combos]


@pytest.mark.parametrize("dataset,model_key,gpu,num_workers", _all_combos, ids=_ids)
def test_budget_positive_and_within_vram(tmp_path, dataset, model_key, gpu, num_workers):
    """Budget must be ≥ 1 and not exceed VRAM capacity."""
    result = _run(tmp_path, dataset=dataset, model_key=model_key,
                  gpu=gpu, num_workers=num_workers)

    assert result.budget >= 1
    bpn = MODEL_PROBES[model_key][0]
    free = GPU_TYPES[gpu]
    max_nodes = int(free * _SAFETY_MARGIN / bpn)
    assert result.budget <= max_nodes, (
        f"budget {result.budget} exceeds mem ceiling {max_nodes}"
    )


@pytest.mark.parametrize("dataset,model_key,gpu,num_workers", _all_combos, ids=_ids)
def test_budget_gives_reasonable_batch_count(tmp_path, dataset, model_key, gpu, num_workers):
    """Budget should translate to ≥ 1 and ≤ 10000 graphs per batch."""
    result = _run(tmp_path, dataset=dataset, model_key=model_key,
                  gpu=gpu, num_workers=num_workers)

    graphs_per_batch = result.budget / result.mean_nodes
    assert graphs_per_batch >= 1, f"budget too small: {graphs_per_batch:.1f} graphs"
    assert graphs_per_batch <= 10000, f"budget too large: {graphs_per_batch:.1f} graphs"


# --- Scaling property tests --------------------------------------------------

@pytest.mark.parametrize("model_key", list(MODEL_PROBES.keys()),
                         ids=[f"{m}_{s}" for m, s in MODEL_PROBES])
def test_larger_gpu_gives_larger_or_equal_budget(tmp_path, model_key):
    """More VRAM → budget should not decrease."""
    budgets = {}
    for gpu in ["v100_16gb", "a100_40gb", "a100_80gb"]:
        result = _run(tmp_path, dataset="set_01", model_key=model_key,
                      gpu=gpu, num_workers=6)
        budgets[gpu] = result.budget

    assert budgets["a100_40gb"] >= budgets["v100_16gb"], (
        f"A100-40 ({budgets['a100_40gb']}) < V100 ({budgets['v100_16gb']})"
    )
    assert budgets["a100_80gb"] >= budgets["a100_40gb"], (
        f"A100-80 ({budgets['a100_80gb']}) < A100-40 ({budgets['a100_40gb']})"
    )


@pytest.mark.parametrize("dataset", list(DATASETS.keys()))
def test_larger_model_gives_smaller_or_equal_budget(tmp_path, dataset):
    """Larger model (more bytes/node) → budget should not increase."""
    pairs = [
        (("vgae", "small"), ("vgae", "large")),
        (("gat", "small"), ("gat", "large")),
        (("dgi", "small"), ("dgi", "large")),
        (("temporal", "small"), ("temporal", "large")),
    ]
    for small_key, large_key in pairs:
        r_small = _run(tmp_path, dataset=dataset, model_key=small_key,
                       gpu="v100_16gb", num_workers=6)
        r_large = _run(tmp_path, dataset=dataset, model_key=large_key,
                       gpu="v100_16gb", num_workers=6)
        assert r_large.budget <= r_small.budget, (
            f"{large_key} budget ({r_large.budget}) > "
            f"{small_key} budget ({r_small.budget}) on {dataset}"
        )


@pytest.mark.parametrize("model_key", list(MODEL_PROBES.keys()),
                         ids=[f"{m}_{s}" for m, s in MODEL_PROBES])
def test_more_workers_does_not_decrease_budget(tmp_path, model_key):
    """More workers → throughput ceiling rises → budget should not shrink."""
    budgets = {}
    for w in [2, 6, 8]:
        result = _run(tmp_path, dataset="set_01", model_key=model_key,
                      gpu="v100_16gb", num_workers=w)
        budgets[w] = result.budget

    assert budgets[6] >= budgets[2], f"w=6 ({budgets[6]}) < w=2 ({budgets[2]})"
    assert budgets[8] >= budgets[6], f"w=8 ({budgets[8]}) < w=6 ({budgets[6]})"


# --- Regime-specific tests ---------------------------------------------------

def test_small_models_are_collation_dominated(tmp_path):
    """Small models with few workers should have cg_ratio > 1."""
    for mk in [("vgae", "small"), ("gat", "small"), ("dgi", "small")]:
        result = _run(tmp_path, dataset="set_01", model_key=mk,
                      gpu="v100_16gb", num_workers=2)
        assert result.cg_ratio is not None and result.cg_ratio > 1.0, (
            f"{mk} with 2 workers: cg_ratio={result.cg_ratio}, expected > 1"
        )


def test_large_temporal_may_be_compute_dominated(tmp_path):
    """Temporal large has high β — with enough workers, GPU becomes bottleneck."""
    result = _run(tmp_path, dataset="set_01", model_key=("temporal", "large"),
                  gpu="v100_16gb", num_workers=8)
    # β=4μs/node, γ=70μs/graph, m̄=28.2, W=8
    # γ_per_node = 70/28.2 = 2.48 μs/node
    # γ_eff = 2.48/8 = 0.31 μs/node
    # ratio = 0.31 / 4.0 = 0.078 → compute-dominated
    assert result.cg_ratio is not None and result.cg_ratio < 1.0
    assert result.throughput_budget is None
    assert result.binding == "memory"


def test_throughput_binds_for_small_model_few_workers(tmp_path):
    """Small model + few workers + big GPU → throughput ceiling < mem ceiling."""
    result = _run(tmp_path, dataset="set_01", model_key=("gat", "small"),
                  gpu="a100_80gb", num_workers=2)
    # On A100-80: mem_budget = 76GB * 0.85 / 800 = ~80M nodes
    # Throughput: gap = 70e-6/2 - 0.08e-6*28.2 = 34.7e-6, B = 0.002/34.7e-6 ≈ 57
    # throughput_budget ≈ 57 * 28.2 ≈ 1617 nodes ≪ 80M
    assert result.throughput_budget is not None
    assert result.throughput_budget < result.mem_budget
    assert result.binding == "throughput"


def test_memory_binds_for_large_model_small_gpu(tmp_path):
    """Large model + small GPU → mem ceiling is tight, binds before throughput."""
    result = _run(tmp_path, dataset="set_02", model_key=("temporal", "large"),
                  gpu="v100_16gb", num_workers=6)
    # mem_budget = 14GB * 0.85 / 9000 ≈ 1.4M nodes
    # This model is compute-dominated (β=4μs) → no throughput budget
    assert result.binding == "memory"


# --- GPS quadratic path (no probe) ------------------------------------------

@pytest.mark.parametrize("gpu", list(GPU_TYPES.keys()))
def test_gps_quadratic_scales_with_sqrt_vram(tmp_path, gpu):
    """GPS budget should scale as sqrt(free_vram)."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text('{"graph_stats":{"node_count":{"mean":28.2}}}')

    free = GPU_TYPES[gpu]
    with (
        patch("graphids.core.preprocessing.budget.cache_dir", return_value=tmp_path),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(free, free)),
    ):
        result = node_budget("set_01", str(tmp_path), conv_type="gps",
                             heads=4, model=None)

    import math
    expected = int(math.sqrt(free / (4 * 3 * 2)))
    assert result.budget == expected
    assert result.binding == "memory"
    assert result.cg_ratio is None


# --- Fallback path (no model) -----------------------------------------------

@pytest.mark.parametrize("dataset", list(DATASETS.keys()))
def test_fallback_consistent_across_datasets(tmp_path, dataset):
    """Without a model, budget depends only on VRAM and mean_nodes (via fallback)."""
    mean_nodes, _ = DATASETS[dataset]
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text(f'{{"graph_stats":{{"node_count":{{"mean":{mean_nodes}}}}}}}')

    with patch("graphids.core.preprocessing.budget.cache_dir", return_value=tmp_path):
        result = node_budget(dataset, str(tmp_path), model=None)

    assert result.binding == "fallback"
    assert result.cg_ratio is None
    assert result.throughput_budget is None
    # All datasets get the same budget because fallback bytes_per_node is constant
    # and VRAM is the same (CPU fallback 12GB)
    from graphids.core.preprocessing.budget import _FALLBACK_BYTES_PER_NODE
    expected = int(12 * 1024**3 * _SAFETY_MARGIN / _FALLBACK_BYTES_PER_NODE)
    assert result.budget == expected
