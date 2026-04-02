"""Throughput-aware node budget for GNN training.

Cost model from docs/reference/gnn_throughput_equations.md.
Computes batch size from regime classification + memory ceiling.

The key insight from the equations: both T_collation and T_gpu scale with batch
size. Their ratio is approximately constant — determined by model architecture,
graph structure, and worker count, NOT batch size. You're either in a
collation-dominated regime (more workers is the fix) or a compute-dominated
regime (fill VRAM). The regime is a property of the system, not a knob to turn.

The one exception: GPU kernel overhead (α) creates a per-step constant cost
that makes very small batches inefficient. When collation-dominated, we cap
the batch to avoid extreme GPU starvation while staying large enough to
amortize kernel overhead.

Usage:
    result = node_budget(dataset, lake_root, model=model,
                         train_dataset=ds, num_workers=6)
    sampler = DynamicBatchSampler(ds, max_num=result.budget, mode="node")
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import structlog
import torch

log = structlog.get_logger()

# --- Constants ---------------------------------------------------------------

# Reserve 15% of free VRAM for allocator fragmentation + edge-density variance.
_SAFETY_MARGIN = 0.85

# Forward-only probe captures activations; multiply to account for gradient
# memory during backward (gradients ≈ activations in size).
_GRAD_MULTIPLIER = 2

# Fallback activation cost per node when no model is available for probing.
_FALLBACK_BYTES_PER_NODE = 32_768

# Conv types with O(N²) global attention (full attention matrix).
_QUADRATIC_CONV_TYPES = frozenset({"gps"})


# --- Cost model (Section 2-4 of gnn_throughput_equations.md) -----------------

@dataclass
class CostCoefficients:
    """Empirically measured rates from a single probe batch.

    These are per-unit rates, NOT total times. Both collation rate and GPU rate
    scale with batch size — their ratio determines the regime.
    """
    collate_per_graph_s: float   # seconds per graph in Batch.from_data_list
    gpu_per_node_s: float        # seconds per node of GPU compute (β)
    gpu_overhead_s: float        # constant GPU overhead per step (α)
    probe_n_graphs: int          # how many graphs the probe used
    probe_n_nodes: int           # total nodes in probe batch


def collation_time(n_graphs: int, coeffs: CostCoefficients) -> float:
    """Predicted collation time for a batch of n_graphs.

    Linear model: T_collate ≈ n_graphs × collate_per_graph.
    Source: Section 2 — O(N_graphs) from PyG Batch.from_data_list.
    """
    return n_graphs * coeffs.collate_per_graph_s


def gpu_time(n_nodes: int, coeffs: CostCoefficients) -> float:
    """Predicted GPU time for a batch of n_nodes.

    Affine model: T_gpu ≈ α + β × n_nodes.
    α = kernel launch overhead (constant per step).
    β = per-node compute cost (scales with model depth × hidden²).
    Source: Section 3 — O(L·h²·(N_E + N_V)).
    """
    return coeffs.gpu_overhead_s + coeffs.gpu_per_node_s * n_nodes


def regime(coeffs: CostCoefficients, num_workers: int, mean_nodes: float) -> str:
    """Classify bottleneck regime from measured coefficients.

    Compares per-node collation rate (γ/W) vs per-node GPU rate (β).
    Source: Section 4 — regime is batch-size-independent when both sides
    scale linearly. The overhead α shifts the balance toward compute-dominated
    for small batches but doesn't change the asymptotic regime.
    """
    # γ = collation cost per node = collate_per_graph / mean_nodes
    gamma = coeffs.collate_per_graph_s / max(1.0, mean_nodes)
    gamma_eff = gamma / max(1, num_workers)
    beta = coeffs.gpu_per_node_s

    if beta <= 0:
        return "collation-dominated"

    ratio = gamma_eff / beta
    if ratio > 2.0:
        return "collation-dominated"
    elif ratio < 0.5:
        return "compute-dominated"
    return "balanced"


# --- Probe -------------------------------------------------------------------

def probe(
    model,
    dataset,
    n_target: int = 2000,
    n_small: int = 200,
    step_fn=None,
) -> tuple[int, CostCoefficients]:
    """Run two probe batches to measure VRAM, collation rate, and GPU rate.

    Probes at two batch sizes (n_small and n_target nodes) to separate the
    affine GPU model T_gpu = α + β·N into overhead (α) and per-node rate (β).

    Returns:
        (bytes_per_node, CostCoefficients)
    """
    from torch_geometric.data import Batch

    # Collect graphs for both probe sizes
    all_graphs, n = [], 0
    small_idx = None
    for g in dataset:
        all_graphs.append(g)
        n += g.num_nodes
        if small_idx is None and n >= n_small:
            small_idx = len(all_graphs)
        if n >= n_target:
            break

    if small_idx is None:
        small_idx = max(1, len(all_graphs) // 4)

    graphs_small = all_graphs[:small_idx]
    graphs_large = all_graphs

    # --- Measure collation time (CPU) on large batch ---
    t0 = time.perf_counter()
    batch_large = Batch.from_data_list(graphs_large)
    t_collate = time.perf_counter() - t0

    # --- Build small batch for two-point GPU measurement ---
    batch_small = Batch.from_data_list(graphs_small)

    batch_large = batch_large.to(model.device)
    batch_small = batch_small.to(model.device)
    nodes_large = batch_large.num_nodes
    nodes_small = batch_small.num_nodes

    was_training = model.training
    model.eval()

    # Warmup both sizes (compile, kernel cache)
    with torch.no_grad():
        (step_fn or model)(batch_small)
        (step_fn or model)(batch_large)
    torch.cuda.synchronize()

    # --- Timed GPU runs ---
    def _time_gpu(batch):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            (step_fn or model)(batch)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    t_gpu_small = _time_gpu(batch_small)
    t_gpu_large = _time_gpu(batch_large)

    # --- VRAM measurement (on large batch, same as before) ---
    torch.cuda.reset_peak_memory_stats(model.device)
    before = torch.cuda.memory_allocated(model.device)
    with torch.no_grad():
        (step_fn or model)(batch_large)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(model.device)

    model.train(was_training)

    del batch_small, batch_large
    torch.cuda.empty_cache()

    fwd_per_node = max(1, int((peak - before) / nodes_large))
    bytes_per_node = fwd_per_node * _GRAD_MULTIPLIER

    # --- Derive affine GPU model: T_gpu = α + β·N ---
    # Two points: (nodes_small, t_gpu_small) and (nodes_large, t_gpu_large)
    if nodes_large > nodes_small:
        beta = max(0.0, (t_gpu_large - t_gpu_small) / (nodes_large - nodes_small))
        alpha = max(0.0, t_gpu_large - beta * nodes_large)
    else:
        beta = t_gpu_large / max(1, nodes_large)
        alpha = 0.0

    n_graphs = len(graphs_large)
    coeffs = CostCoefficients(
        collate_per_graph_s=t_collate / n_graphs,
        gpu_per_node_s=beta,
        gpu_overhead_s=alpha,
        probe_n_graphs=n_graphs,
        probe_n_nodes=nodes_large,
    )

    log.info("budget_probe",
             bytes_per_node=bytes_per_node,
             probe_nodes_small=nodes_small,
             probe_nodes_large=nodes_large,
             probe_graphs=n_graphs,
             t_collate_ms=round(t_collate * 1000, 1),
             t_gpu_small_ms=round(t_gpu_small * 1000, 1),
             t_gpu_large_ms=round(t_gpu_large * 1000, 1),
             alpha_ms=round(alpha * 1000, 2),
             beta_us_per_node=round(beta * 1e6, 3),
             collate_per_graph_us=round(coeffs.collate_per_graph_s * 1e6, 1),
             method="step_fn" if step_fn else "forward")

    return bytes_per_node, coeffs


# --- Budget computation ------------------------------------------------------

@dataclass
class BudgetResult:
    """Structured result from node_budget(). All fields logged for analysis."""
    budget: int                       # actual max_num for DynamicBatchSampler
    mean_nodes: float                 # from cache_metadata.json
    mem_budget: int                   # VRAM ceiling (nodes)
    throughput_budget: int | None     # pipeline ceiling (nodes), None if no probe
    binding: str                      # "memory" | "throughput" | "fallback"
    regime: str                       # "collation-dominated" | "compute-dominated" | "balanced"
    coefficients: CostCoefficients | None = field(default=None, repr=False)


def _throughput_budget_nodes(
    coeffs: CostCoefficients,
    num_workers: int,
    mean_nodes: float,
) -> int | None:
    """Compute the batch size where pipeline delivery matches GPU consumption.

    Solves: T_collation(B) / W = T_gpu(B)
    Where T_collation = γ·B (linear in graphs)
    And   T_gpu = α + β·N = α + β·B·mean_nodes (affine in graphs via nodes)

    Solution: B = α / (γ/W - β·mean_nodes)
    Exists only when γ/W > β·mean_nodes (collation-dominated regime).

    When it exists, this is the batch size (in graphs) where the pipeline
    exactly keeps up with the GPU. Below this, the GPU starves less but
    wastes kernel overhead. Above this, the GPU starves.
    """
    gamma = coeffs.collate_per_graph_s
    beta_per_graph = coeffs.gpu_per_node_s * mean_nodes
    alpha = coeffs.gpu_overhead_s
    delivery_rate = gamma / max(1, num_workers)

    # Only solvable when collation per graph exceeds GPU per graph
    gap = delivery_rate - beta_per_graph
    if gap <= 0:
        return None  # compute-dominated — memory binds

    if alpha <= 0:
        # Pure linear model: ratio is constant, no finite optimum.
        # Use mem_budget (regime check handles this).
        return None

    optimal_graphs = max(1, int(alpha / gap))
    return int(optimal_graphs * mean_nodes)


def node_budget(
    dataset: str,
    lake_root: str,
    *,
    conv_type: str = "gatv2",
    heads: int = 4,
    model=None,
    train_dataset=None,
    num_workers: int = 2,
) -> BudgetResult:
    """Compute node budget from regime classification + memory ceiling.

    Memory ceiling: largest batch that fits in VRAM without OOM.
    Regime classification: determines if the pipeline is collation-dominated
    or compute-dominated. In the collation-dominated regime AND when the GPU
    has measurable per-step overhead (α), computes an optimal batch size that
    balances kernel overhead amortization against pipeline delivery rate.

    Args:
        dataset: catalog name (e.g. "set_01")
        lake_root: path to experiment lake
        conv_type: convolution type (affects quadratic budget path)
        heads: attention heads (for quadratic budget)
        model: LightningModule on device (enables live probe)
        train_dataset: dataset for probe batch
        num_workers: DataLoader worker count
    """
    from graphids.config import cache_dir

    metadata_path = cache_dir(lake_root, dataset) / "cache_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"cache_metadata.json not found at {metadata_path}. "
            "Run preprocessing first."
        )
    stats = json.loads(metadata_path.read_text())["graph_stats"]["node_count"]
    mean_nodes = stats["mean"]

    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
    else:
        free = 12 * 1024**3  # CPU fallback for testing

    # Quadratic conv types have O(N²) attention — separate path.
    if conv_type in _QUADRATIC_CONV_TYPES:
        budget = int(math.sqrt(free / (heads * 3 * 2)))
        log.info("node_budget", conv_type=conv_type, budget=budget,
                 free_vram_gb=round(free / 1e9, 2), method="quadratic",
                 binding="memory")
        return BudgetResult(
            budget=budget, mean_nodes=mean_nodes,
            mem_budget=budget, throughput_budget=None, binding="memory",
            regime="compute-dominated",
        )

    # --- Run probe if model available ---
    coeffs = None
    bytes_per_node = _FALLBACK_BYTES_PER_NODE

    if model is not None and train_dataset is not None and torch.cuda.is_available():
        step_fn = getattr(model, "_step", None)
        bytes_per_node, coeffs = probe(model, train_dataset, step_fn=step_fn)

    # --- Memory ceiling ---
    mem_budget = int(free * _SAFETY_MARGIN / bytes_per_node)

    # --- Regime classification + throughput budget ---
    throughput_budget = None
    detected_regime = "unknown"

    if coeffs is not None:
        detected_regime = regime(coeffs, num_workers, mean_nodes)
        throughput_budget = _throughput_budget_nodes(coeffs, num_workers, mean_nodes)

    # --- Final budget ---
    if throughput_budget is not None and throughput_budget < mem_budget:
        budget = throughput_budget
        binding = "throughput"
    else:
        budget = mem_budget
        binding = "fallback" if coeffs is None else "memory"

    budget = max(1, budget)

    log.info("node_budget",
             budget=budget,
             mem_budget=mem_budget,
             throughput_budget=throughput_budget,
             binding=binding,
             regime=detected_regime,
             num_workers=num_workers,
             free_vram_gb=round(free / 1e9, 2),
             bytes_per_node=bytes_per_node,
             mean_nodes=round(mean_nodes, 1),
             alpha_ms=round(coeffs.gpu_overhead_s * 1000, 2) if coeffs else None,
             beta_us=round(coeffs.gpu_per_node_s * 1e6, 3) if coeffs else None)

    return BudgetResult(
        budget=budget,
        mean_nodes=mean_nodes,
        mem_budget=mem_budget,
        throughput_budget=throughput_budget,
        binding=binding,
        regime=detected_regime,
        coefficients=coeffs,
    )
