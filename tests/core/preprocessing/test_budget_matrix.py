"""Budget matrix tests: exercise node_budget() across real project configurations.

Parametrized over datasets, model types/scales, and GPU types using realistic
VRAM probe values measured on Pitzer V100. These are CPU tests (mock
_probe_vram) — actual GPU validation is a separate SLURM job.
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

# --- Realistic VRAM probe values per model type + scale ----------------------
# Measured on Pitzer V100 (job 46275772, 2026-04-03). Median across datasets.
# bytes_per_node includes backward multiplier.

MODEL_PROBES = {
    # (model_type, scale): (bytes_per_node, bwd_mult, conv_type)
    ("vgae", "small"):     (34620,  1.39, "gatv2"),
    ("vgae", "large"):     (50100,  1.26, "gatv2"),
    ("gat", "small"):      (59860,  1.29, "gatv2"),
    ("gat", "large"):      (223740, 1.52, "gatv2"),
    ("dgi", "small"):      (13970,  2.0,  "gatv2"),
    ("dgi", "large"):      (80140,  2.0,  "gatv2"),
}

# --- GPU free VRAM after model load (from clusters.yaml) --------------------

_clusters = read_yaml(CONFIG_DIR / "resources" / "clusters.yaml")
GPU_TYPES = {
    name: int(spec["free_gb"] * 1024**3)
    for name, spec in _clusters["clusters"]["gpu_vram"].items()
}


# --- Helpers -----------------------------------------------------------------

def _run(tmp_path, *, dataset, model_key, gpu):
    """Run node_budget with mocked VRAM probe."""
    mean_nodes, p95_nodes = DATASETS[dataset]
    bpn, bwd_mult, conv_type = MODEL_PROBES[model_key]
    free = GPU_TYPES[gpu]

    edge_mean = mean_nodes * 4.5
    edge_p95 = p95_nodes * 4.5
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text(
        f'{{"graph_stats":{{"node_count":{{"mean":{mean_nodes},"p95":{p95_nodes}}},'
        f'"edge_count":{{"mean":{edge_mean},"p95":{edge_p95}}}}}}}'
    )

    def mock_probe_vram(model, ds, step_fn=None):
        return bpn, bwd_mult

    with (
        patch("graphids.core.preprocessing.budget.cache_dir", return_value=tmp_path),
        patch("graphids.core.preprocessing.budget._probe_vram", mock_probe_vram),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(free, free)),
    ):
        return node_budget(
            dataset, str(tmp_path), conv_type=conv_type,
            model=True, train_dataset=True,
        )


# --- Parametrized tests ------------------------------------------------------

_all_combos = [
    (ds, mk, gpu)
    for ds in DATASETS
    for mk in MODEL_PROBES
    for gpu in GPU_TYPES
]

_ids = [f"{ds}-{mk[0]}_{mk[1]}-{gpu}" for ds, mk, gpu in _all_combos]


@pytest.mark.parametrize("dataset,model_key,gpu", _all_combos, ids=_ids)
def test_budget_positive_and_within_vram(tmp_path, dataset, model_key, gpu):
    """Budget must be >= 1 and not exceed VRAM capacity."""
    result = _run(tmp_path, dataset=dataset, model_key=model_key, gpu=gpu)

    assert result.budget >= 1
    bpn = MODEL_PROBES[model_key][0]
    free = GPU_TYPES[gpu]
    mean_nodes, p95_nodes = DATASETS[dataset]
    mean_epn = (mean_nodes * 4.5) / mean_nodes
    p95_epn = (p95_nodes * 4.5) / p95_nodes
    edge_ratio = max(1.0, p95_epn / mean_epn)
    effective_bpn = int(bpn * edge_ratio)
    max_nodes = int(free * _SAFETY_MARGIN / effective_bpn)
    assert result.budget <= max_nodes, (
        f"budget {result.budget} exceeds mem ceiling {max_nodes}"
    )


@pytest.mark.parametrize("dataset,model_key,gpu", _all_combos, ids=_ids)
def test_budget_gives_reasonable_batch_count(tmp_path, dataset, model_key, gpu):
    """Budget should translate to >= 1 graphs and fit in VRAM."""
    result = _run(tmp_path, dataset=dataset, model_key=model_key, gpu=gpu)

    graphs_per_batch = result.budget / result.mean_nodes
    assert graphs_per_batch >= 1, f"budget too small: {graphs_per_batch:.1f} graphs"
    assert graphs_per_batch <= 1_000_000, f"budget too large: {graphs_per_batch:.1f} graphs"


# --- Scaling property tests --------------------------------------------------

@pytest.mark.parametrize("model_key", list(MODEL_PROBES.keys()),
                         ids=[f"{m}_{s}" for m, s in MODEL_PROBES])
def test_larger_gpu_gives_larger_or_equal_budget(tmp_path, model_key):
    """More VRAM -> budget should not decrease."""
    budgets = {}
    for gpu in ["v100_16gb", "a100_40gb", "a100_80gb"]:
        result = _run(tmp_path, dataset="set_01", model_key=model_key, gpu=gpu)
        budgets[gpu] = result.budget

    assert budgets["a100_40gb"] >= budgets["v100_16gb"], (
        f"A100-40 ({budgets['a100_40gb']}) < V100 ({budgets['v100_16gb']})"
    )
    assert budgets["a100_80gb"] >= budgets["a100_40gb"], (
        f"A100-80 ({budgets['a100_80gb']}) < A100-40 ({budgets['a100_40gb']})"
    )


@pytest.mark.parametrize("dataset", list(DATASETS.keys()))
def test_larger_model_gives_smaller_or_equal_mem_budget(tmp_path, dataset):
    """Larger model (more bytes/node) -> memory ceiling should not increase."""
    pairs = [
        (("vgae", "small"), ("vgae", "large")),
        (("gat", "small"), ("gat", "large")),
        (("dgi", "small"), ("dgi", "large")),
    ]
    for small_key, large_key in pairs:
        r_small = _run(tmp_path, dataset=dataset, model_key=small_key, gpu="v100_16gb")
        r_large = _run(tmp_path, dataset=dataset, model_key=large_key, gpu="v100_16gb")
        assert r_large.mem_budget <= r_small.mem_budget, (
            f"{large_key} mem_budget ({r_large.mem_budget}) > "
            f"{small_key} mem_budget ({r_small.mem_budget}) on {dataset}"
        )


def test_budget_always_memory_bound(tmp_path):
    """Budget equals mem_budget for all combos (no throughput floor)."""
    for mk in MODEL_PROBES:
        for gpu in GPU_TYPES:
            result = _run(tmp_path, dataset="set_01", model_key=mk, gpu=gpu)
            assert result.budget == result.mem_budget, (
                f"{mk} on {gpu}: budget {result.budget} != ceiling {result.mem_budget}"
            )
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


# --- Fallback path (no model) -----------------------------------------------

@pytest.mark.parametrize("dataset", list(DATASETS.keys()))
def test_fallback_consistent_across_datasets(tmp_path, dataset):
    """Without a model, budget depends only on VRAM (fallback bytes_per_node)."""
    mean_nodes, _ = DATASETS[dataset]
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text(f'{{"graph_stats":{{"node_count":{{"mean":{mean_nodes}}}}}}}')

    with patch("graphids.core.preprocessing.budget.cache_dir", return_value=tmp_path):
        result = node_budget(dataset, str(tmp_path), model=None)

    assert result.binding == "fallback"
    from graphids.core.preprocessing.budget import _FALLBACK_BYTES_PER_NODE
    expected = int(12 * 1024**3 * _SAFETY_MARGIN / _FALLBACK_BYTES_PER_NODE)
    assert result.budget == expected
