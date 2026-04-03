"""Budget matrix tests: exercise node_budget() across real project configurations.

Parametrized over datasets, model types/scales, GPU types, and worker counts
using realistic probe values estimated from model architectures. These are
CPU tests (mock _probe) — actual GPU validation is a separate SLURM job.

To generate the full budget profile artifact, run:
    python -m graphids probe-budget --matrix
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from graphids.config import CONFIG_DIR
from graphids.config.yaml_utils import read_yaml
from graphids.core.preprocessing.budget import (
    _SAFETY_MARGIN,
    BudgetResult,
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
    # (model_type, scale): (bytes_per_node, bwd_mult, gamma_s, alpha_s, beta_s, conv_type)
    # Measured on Pitzer V100 (job 46275772, 2026-04-03). Median across datasets.
    # γ fixed: cuda.synchronize + 3-sample median eliminated DGI anomaly (200ms → 42μs).
    # bytes_per_node includes backward multiplier. Temporal probes fail (windowed batching).
    #
    # VGAE small: ~26K params, bwd_mult=1.39
    ("vgae", "small"):    (34620,  1.39, 40e-6, 0.008, 0.11e-6, "gatv2"),
    # VGAE large: ~430K params, bwd_mult=1.26
    ("vgae", "large"):    (50100,  1.26, 42e-6, 0.007, 0.15e-6, "gatv2"),
    # GAT small: ~190K params, bwd_mult=1.29
    ("gat", "small"):     (59860,  1.29, 40e-6, 0.003, 0.86e-6, "gatv2"),
    # GAT large: ~2.5M params, bwd_mult=1.52
    ("gat", "large"):     (223740, 1.52, 42e-6, 0.005, 0.73e-6, "gatv2"),
    # DGI small: ~17K params, bwd_mult=2.0 (fallback — backward probe fails)
    ("dgi", "small"):     (13970,  2.0, 40e-6, 0.007, 0.03e-6, "gatv2"),
    # DGI large: ~345K params, bwd_mult=2.0 (fallback)
    ("dgi", "large"):     (80140,  2.0, 42e-6, 0.006, 0.10e-6, "gatv2"),
    # Temporal small: ~254K params (probe fails — needs windowed batching, estimates only)
    ("temporal", "small"): (100000, 2.0, 42e-6, 0.008, 1.50e-6, "gatv2"),
    # Temporal large: ~10.2M params (probe fails — estimates only)
    ("temporal", "large"): (250000, 2.0, 42e-6, 0.015, 4.00e-6, "gatv2"),
}

# --- GPU free VRAM after model load (from clusters.yaml) --------------------

_clusters = read_yaml(CONFIG_DIR / "resources" / "clusters.yaml")
GPU_TYPES = {
    name: int(spec["free_gb"] * 1024**3)
    for name, spec in _clusters["clusters"]["gpu_vram"].items()
}

WORKER_COUNTS = [2, 6, 8]


# --- Helpers -----------------------------------------------------------------

def _run(tmp_path, *, dataset, model_key, gpu, num_workers):
    """Run node_budget with mocked probe and VRAM."""
    mean_nodes, p95_nodes = DATASETS[dataset]
    bpn, bwd_mult, gamma, alpha, beta, conv_type = MODEL_PROBES[model_key]
    free = GPU_TYPES[gpu]

    # Include edge_count stats for edge-aware margin.
    # Estimate: edges ≈ 4.5 × nodes (typical CAN bus graph density).
    # p95/mean ratio ≈ 1.05 (near-constant E/N for CAN bus).
    edge_mean = mean_nodes * 4.5
    edge_p95 = p95_nodes * 4.5
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text(
        f'{{"graph_stats":{{"node_count":{{"mean":{mean_nodes},"p95":{p95_nodes}}},'
        f'"edge_count":{{"mean":{edge_mean},"p95":{edge_p95}}}}}}}'
    )

    def mock_probe(model, ds, step_fn=None):
        return bpn, bwd_mult, gamma, alpha, beta

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
    # Edge-aware margin: effective_bpn = bpn × (p95_epn / mean_epn).
    # Mock metadata has edge/node ratio = 4.5 for both mean and p95,
    # so p95_epn/mean_epn ≈ 1.0 (near-constant density) → no adjustment.
    # For datasets where p95_nodes > mean_nodes, ratio can be slightly > 1.
    mean_nodes, p95_nodes = DATASETS[dataset]
    mean_epn = (mean_nodes * 4.5) / mean_nodes  # = 4.5
    p95_epn = (p95_nodes * 4.5) / p95_nodes     # = 4.5
    edge_ratio = max(1.0, p95_epn / mean_epn)   # = 1.0
    effective_bpn = int(bpn * edge_ratio)
    max_nodes = int(free * _SAFETY_MARGIN / effective_bpn)
    assert result.budget <= max_nodes, (
        f"budget {result.budget} exceeds mem ceiling {max_nodes}"
    )


@pytest.mark.parametrize("dataset,model_key,gpu,num_workers", _all_combos, ids=_ids)
def test_budget_gives_reasonable_batch_count(tmp_path, dataset, model_key, gpu, num_workers):
    """Budget should translate to ≥ 1 graphs and fit in VRAM.

    Upper bound is generous: small CAN bus graphs (21 nodes) on H100 (90GB)
    can legitimately yield 200K+ graphs/batch. The VRAM ceiling test is the
    real safety check; this test catches orders-of-magnitude errors.
    """
    result = _run(tmp_path, dataset=dataset, model_key=model_key,
                  gpu=gpu, num_workers=num_workers)

    graphs_per_batch = result.budget / result.mean_nodes
    assert graphs_per_batch >= 1, f"budget too small: {graphs_per_batch:.1f} graphs"
    assert graphs_per_batch <= 1_000_000, f"budget too large: {graphs_per_batch:.1f} graphs"


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
def test_larger_model_gives_smaller_or_equal_mem_budget(tmp_path, dataset):
    """Larger model (more bytes/node) → memory ceiling should not increase.

    Compares mem_budget (VRAM ceiling) not final budget, because binding
    regime can switch: a small throughput-bound model can have a tighter
    final budget than a large memory-bound model.
    """
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
        assert r_large.mem_budget <= r_small.mem_budget, (
            f"{large_key} mem_budget ({r_large.mem_budget}) > "
            f"{small_key} mem_budget ({r_small.mem_budget}) on {dataset}"
        )


@pytest.mark.parametrize("model_key", list(MODEL_PROBES.keys()),
                         ids=[f"{m}_{s}" for m, s in MODEL_PROBES])
def test_more_workers_does_not_decrease_budget(tmp_path, model_key):
    """Budget (memory ceiling) should not change with worker count."""
    budgets = {}
    for w in [2, 6, 8]:
        result = _run(tmp_path, dataset="set_01", model_key=model_key,
                      gpu="v100_16gb", num_workers=w)
        budgets[w] = result.budget

    assert budgets[6] >= budgets[2], f"w=6 ({budgets[6]}) < w=2 ({budgets[2]})"
    assert budgets[8] >= budgets[6], f"w=8 ({budgets[8]}) < w=6 ({budgets[6]})"


# --- Regime-specific tests ---------------------------------------------------

def test_small_models_are_collation_dominated(tmp_path):
    """Small models with few workers should have cg_ratio > 1 (except GAT).

    GAT small is compute-bound even at W=2 (cg_ratio ≈ 0.64) because β is
    high relative to γ. VGAE and DGI have near-zero β → collation-bound.
    """
    for mk in [("vgae", "small"), ("dgi", "small")]:
        result = _run(tmp_path, dataset="set_01", model_key=mk,
                      gpu="v100_16gb", num_workers=2)
        assert result.cg_ratio is not None and result.cg_ratio > 1.0, (
            f"{mk} with 2 workers: cg_ratio={result.cg_ratio}, expected > 1"
        )

    # GAT small is compute-bound — GPU is bottleneck, not collation
    result = _run(tmp_path, dataset="set_01", model_key=("gat", "small"),
                  gpu="v100_16gb", num_workers=2)
    assert result.cg_ratio is not None and result.cg_ratio < 1.0, (
        f"gat/small with 2 workers: cg_ratio={result.cg_ratio}, expected < 1"
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
    assert result.binding == "memory"


def test_budget_always_memory_bound(tmp_path):
    """Budget equals mem_budget — throughput floor never exceeds VRAM ceiling."""
    # Small model on big GPU
    r1 = _run(tmp_path, dataset="set_01", model_key=("gat", "small"),
              gpu="a100_80gb", num_workers=2)
    assert r1.budget == r1.mem_budget
    assert r1.binding == "memory"

    # Large model on small GPU
    r2 = _run(tmp_path, dataset="set_02", model_key=("temporal", "large"),
              gpu="v100_16gb", num_workers=6)
    assert r2.budget == r2.mem_budget
    assert r2.binding == "memory"


def test_throughput_floor_never_exceeds_budget(tmp_path):
    """Budget must not exceed mem_budget, even when floor > ceiling.

    When floor > ceiling (e.g. DGI large on V100), GPU overhead can't be
    fully amortized within VRAM. Budget uses mem_budget and logs a warning.
    """
    for mk in MODEL_PROBES:
        for gpu in GPU_TYPES:
            result = _run(tmp_path, dataset="set_01", model_key=mk,
                          gpu=gpu, num_workers=6)
            assert result.budget <= result.mem_budget, (
                f"{mk} on {gpu}: budget {result.budget} > ceiling {result.mem_budget}"
            )
            assert result.binding == "memory" or result.binding == "fallback"


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
    # All datasets get the same budget because fallback bytes_per_node is constant
    # and VRAM is the same (CPU fallback 12GB)
    from graphids.core.preprocessing.budget import _FALLBACK_BYTES_PER_NODE
    expected = int(12 * 1024**3 * _SAFETY_MARGIN / _FALLBACK_BYTES_PER_NODE)
    assert result.budget == expected
