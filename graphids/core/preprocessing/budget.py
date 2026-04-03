"""Node budget for PyG's DynamicBatchSampler.

    budget = clamp(throughput_floor, mem_ceiling)

Memory ceiling: max nodes that fit in VRAM (hard upper bound — OOM above).
Throughput floor: minimum batch size to amortize GPU kernel overhead α.
    Below the floor, GPU idles between kernel launches. Above it,
    throughput is flat (collation-bound) or still increasing (compute-bound).
    Only exists when collation is slower than GPU (γ/W > β·m̄).

For all current models, mem_ceiling >> floor, so budget = mem_ceiling.
The floor is a guard against manual batch-size reduction or future models
with large α.

Cost model: docs/reference/gnn_throughput_equations.md
Audit: docs/reference/budget-cost-model-audit.md
Each constant is tagged DERIVED / HEURISTIC / FALLBACK.
"""

from __future__ import annotations

import gc
import json
import math
import statistics
import time
from dataclasses import dataclass

from graphids.log import get_logger
import torch
from torch.utils.benchmark import Timer as BenchmarkTimer

from graphids.config import cache_dir

log = get_logger(__name__)

# HEURISTIC: 15% VRAM reserve for allocator fragmentation. Variable-size
# graph batches have higher memory variance than fixed-size, so wider than
# Lightning's 5%. Standard range in PyTorch training code is 10-20%.
_SAFETY_MARGIN = 0.85

# HEURISTIC: probe measures forward-only memory. Training adds gradients
# (≈ activations) + optimizer state (small for GNNs with 24K-200K params).
# 2× is a rough upper bound — could be 1.5-3× depending on model.
# Source: Chen et al. 2016, popXL Tutorial 2.
_GRAD_MULTIPLIER = 2

# FALLBACK: conservative bytes/node when no model is available for probing.
# Real models use 1-8KB/node. Overestimating is safe (smaller budget, no OOM).
_FALLBACK_BYTES_PER_NODE = 32_768

# DERIVED: conv types with O(N²) global attention (not O(E) over edges).
_QUADRATIC_CONV_TYPES = frozenset({"gps"})


@dataclass
class BudgetResult:
    """Output of node_budget()."""
    budget: int                  # max_num for DynamicBatchSampler
    mean_nodes: float            # dataset mean from cache_metadata.json
    mem_budget: int              # VRAM ceiling in nodes
    throughput_floor: int | None  # minimum batch nodes to amortize α (None = no floor)
    binding: str                 # "memory" | "throughput_floor" | "fallback"
    cg_ratio: float | None       # (γ/W)/(β·bwd_mult) — >1 = collation bottleneck (training-adjusted)
    # Raw probe measurements (for post-hoc analysis)
    bytes_per_node: int | None = None
    gamma_us: float | None = None   # γ: collation cost per graph (μs)
    alpha_ms: float | None = None   # α: GPU overhead per step (ms)
    beta_us: float | None = None    # β: GPU cost per node (μs)
    # Edge-aware margin (Phase 1B)
    edges_per_node_p95: float | None = None   # p95 edge/node ratio from cache stats
    # Backward multiplier (Phase 2A)
    backward_multiplier: float | None = None  # measured fwd+bwd / fwd ratio
    # KD teacher (Phase 2B)
    teacher_vram_bytes: int = 0               # estimated teacher VRAM reservation
    # Compile status (Phase 2C)
    is_compiled: bool | None = None           # torch.compile status when probed


def _extract_loss(output):
    """Handle _step() return formats: scalar, tuple, or dict."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        return output["loss"]
    raise TypeError(f"Cannot extract loss from {type(output)}")


def _probe(model, dataset, step_fn=None) -> tuple[int, float, float, float, float]:
    """Measure bytes_per_node (with backward multiplier), γ, α, β.

    VRAM: single-point measurement on large batch, scaled by measured
    backward multiplier (fwd+bwd peak / fwd peak). Falls back to
    _GRAD_MULTIPLIER when backward measurement fails.

    Timing: fits  T_gpu(N) = α + β·N  from two-point measurement.

    Returns (bytes_per_node, backward_multiplier, gamma, alpha, beta).
    bytes_per_node includes the backward multiplier.
    """
    from torch_geometric.data import Batch

    # --- Collect graphs into two probe batches ---
    # HEURISTIC: 2000 nodes for large batch (~70 graphs at mean 28 nodes),
    # 200 nodes for small batch (~7 graphs). Large enough for reliable timing,
    # small enough to be fast during DataLoader setup.
    N_TARGET, N_SMALL = 2000, 200

    all_graphs, total_nodes = [], 0
    small_idx = None
    for g in dataset:
        all_graphs.append(g)
        total_nodes += g.num_nodes
        if small_idx is None and total_nodes >= N_SMALL:
            small_idx = len(all_graphs)
        if total_nodes >= N_TARGET:
            break

    # FALLBACK: if dataset is too small, use first 25% as small batch.
    # Arbitrary — just needs to differ from large batch for the two-point solve.
    if small_idx is None:
        small_idx = max(1, len(all_graphs) // 4)

    graphs_small = all_graphs[:small_idx]
    graphs_large = all_graphs

    # --- γ: collation rate (CPU, 3-sample median) ---
    # Pure CPU work — but CUDA context init can stall the CPU thread.
    # Flush GPU work + GC before timing to avoid contamination.
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    gc.collect()

    gamma_samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        batch_large = Batch.from_data_list(graphs_large)
        gamma_samples.append((time.perf_counter() - t0) / len(graphs_large))
    gamma = statistics.median(gamma_samples)  # seconds per graph

    batch_small = Batch.from_data_list(graphs_small)
    batch_large = batch_large.to(model.device)
    batch_small = batch_small.to(model.device)
    nodes_large = batch_large.num_nodes
    nodes_small = batch_small.num_nodes

    was_training = model.training
    model.eval()

    # Warmup: torch.compile, kernel JIT, cuDNN autotuning on both sizes.
    fn = step_fn or model
    with torch.no_grad():
        fn(batch_small)
        fn(batch_large)
    if model.device.type == "cuda":
        torch.cuda.synchronize()

    # --- T_gpu at two batch sizes (BenchmarkTimer: multi-sample median) ---
    def _make_fwd(batch):
        def _fwd():
            with torch.no_grad():
                fn(batch)
        return _fwd

    t_gpu_small = BenchmarkTimer(
        stmt="_fwd()", globals={"_fwd": _make_fwd(batch_small)},
    ).blocked_autorange(min_run_time=0.2).median

    t_gpu_large = BenchmarkTimer(
        stmt="_fwd()", globals={"_fwd": _make_fwd(batch_large)},
    ).blocked_autorange(min_run_time=0.2).median

    # --- Forward-only VRAM from one forward pass ---
    # Single-point measurement: bytes_per_node = peak_delta / num_nodes.
    # The 2×2 node+edge decomposition (vram = A·N + B·E) is ill-conditioned
    # for CAN bus graphs because E/N ratio is near-constant across batches,
    # making the determinant ~0. Edge-aware margin is applied in node_budget()
    # using edge_count stats from cache_metadata.json instead.
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        before = torch.cuda.memory_allocated(model.device)
        with torch.no_grad():
            fn(batch_large)
        torch.cuda.synchronize()
        vram_large = torch.cuda.max_memory_allocated(model.device) - before
    else:
        vram_large = 0

    fwd_per_node = max(1, int(vram_large / max(1, nodes_large))) if vram_large > 0 else 1

    # --- Backward pass multiplier ---
    # Measures real fwd+bwd peak vs forward-only peak. This runs outside
    # Lightning's training loop (during DataLoader setup), so we use
    # torch.autograd.backward directly — Lightning's Trainer isn't active here.
    backward_multiplier = _GRAD_MULTIPLIER
    if model.device.type == "cuda" and step_fn is not None:
        try:
            model.train()
            batch_bwd = batch_large.clone()
            torch.cuda.reset_peak_memory_stats(model.device)
            before = torch.cuda.memory_allocated(model.device)
            loss = _extract_loss(step_fn(batch_bwd))
            torch.autograd.backward(loss)
            torch.cuda.synchronize()
            bwd_peak = torch.cuda.max_memory_allocated(model.device) - before
            if vram_large > 0:
                backward_multiplier = max(1.0, bwd_peak / vram_large)
            model.zero_grad(set_to_none=True)
            del batch_bwd, loss
            model.eval()
        except Exception as exc:
            log.warning("backward_probe_failed", fallback=_GRAD_MULTIPLIER,
                        error=str(exc)[:120])
            model.eval()

    # Apply backward multiplier to get training-time cost per node
    bytes_per_node = int(fwd_per_node * backward_multiplier)

    model.train(was_training)
    del batch_small, batch_large
    if model.device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Solve T_gpu = α + β·N ---
    #   β = (T₂ - T₁) / (N₂ - N₁)
    #   α = T₂ - β·N₂
    # Clamped ≥ 0: negative = noise > signal. Biases toward zero which makes
    # throughput_budget larger or None — safe, memory ceiling still protects.
    if nodes_large > nodes_small:
        beta = max(0.0, (t_gpu_large - t_gpu_small) / (nodes_large - nodes_small))
        alpha = max(0.0, t_gpu_large - beta * nodes_large)
    else:
        beta = t_gpu_large / max(1, nodes_large)
        alpha = 0.0

    log.info("budget_probe",
             bytes_per_node=bytes_per_node,
             backward_multiplier=round(backward_multiplier, 2),
             fwd_per_node=fwd_per_node,
             nodes_small=nodes_small, nodes_large=nodes_large,
             n_graphs=len(graphs_large),
             gamma_median_us=round(gamma * 1e6, 1),
             t_gpu_small_ms=round(t_gpu_small * 1000, 1),
             t_gpu_large_ms=round(t_gpu_large * 1000, 1),
             alpha_ms=round(alpha * 1000, 2),
             beta_us=round(beta * 1e6, 3),
             gamma_us=round(gamma * 1e6, 1))

    return bytes_per_node, backward_multiplier, gamma, alpha, beta


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
    """Compute max_num for DynamicBatchSampler(mode="node").

    1. Read mean_nodes from cache_metadata.json.
    2. Read free VRAM (after model load, before optimizer).
    3. GPS conv → quadratic VRAM formula, return early.
    4. If model available → _probe() for bytes_per_node, γ, α, β.
    5. mem_budget = free × SAFETY_MARGIN / bytes_per_node.
    6. throughput_floor = α·m̄ / (γ/W − β_train·m̄) [if collation-bound].
    7. budget = clamp(throughput_floor, mem_budget).
    """
    # --- Step 1: dataset statistics ---
    metadata_path = cache_dir(lake_root, dataset) / "cache_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"cache_metadata.json not found at {metadata_path}. "
            "Run preprocessing first."
        )
    graph_stats = json.loads(metadata_path.read_text())["graph_stats"]
    node_stats = graph_stats["node_count"]
    edge_stats = graph_stats.get("edge_count")  # may be absent in old caches
    mean_nodes = node_stats["mean"]

    # --- Step 2: free VRAM + KD teacher reservation ---
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
    else:
        free = 12 * 1024**3  # FALLBACK: 12GB for CPU-only testing

    # Subtract KD teacher footprint if present (Phase 2B)
    teacher_vram = 0
    if model is not None and hasattr(model, "teacher") and model.teacher is not None:
        teacher_params = sum(p.numel() * p.element_size() for p in model.teacher.parameters())
        # Inference activations ~2.5× params (no gradients, but intermediate tensors)
        teacher_vram = int(teacher_params * 2.5)
        log.info("kd_teacher_vram", bytes=teacher_vram, mb=round(teacher_vram / 1e6, 1))
    effective_free = free - teacher_vram

    # --- Step 2b: compile status (Phase 2C) ---
    is_compiled = hasattr(model, "_orig_mod") if model is not None else None

    # --- Step 3: quadratic conv types ---
    # DERIVED: GPS global attention allocates [N×N×K] per head.
    # Memory ≈ N² × heads × 3 (Q,K,V) × 2 bytes (fp16).
    # Solve: N ≤ sqrt(free / (heads × 3 × 2))
    if conv_type in _QUADRATIC_CONV_TYPES:
        budget = int(math.sqrt(effective_free / (heads * 3 * 2)))
        log.info("node_budget", conv_type=conv_type, budget=budget,
                 free_vram_gb=round(effective_free / 1e9, 2), binding="memory")
        return BudgetResult(
            budget=budget, mean_nodes=mean_nodes,
            mem_budget=budget, throughput_floor=None,
            binding="memory", cg_ratio=None,
            teacher_vram_bytes=teacher_vram, is_compiled=is_compiled,
        )

    # --- Step 4: probe ---
    gamma = alpha = beta = None
    bytes_per_node = _FALLBACK_BYTES_PER_NODE
    backward_multiplier = None

    if model is not None and train_dataset is not None and torch.cuda.is_available():
        step_fn = getattr(model, "_step", None)
        bytes_per_node, backward_multiplier, gamma, alpha, beta = (
            _probe(model, train_dataset, step_fn=step_fn)
        )

    # --- Step 5: memory ceiling ---
    # Edge-aware margin: the probe measures VRAM at the dataset's average E/N
    # ratio (baked into the batch). Batches with denser graphs use more VRAM.
    # Scale bytes_per_node by (p95_edges/p95_nodes) / (mean_edges/mean_nodes)
    # to account for worst-case batches.
    edges_per_node_p95 = None
    effective_bpn = bytes_per_node
    if edge_stats is not None and "p95" in edge_stats and "p95" in node_stats:
        mean_epn = edge_stats["mean"] / max(1.0, node_stats["mean"])
        p95_epn = edge_stats["p95"] / max(1.0, node_stats["p95"])
        edges_per_node_p95 = p95_epn
        if mean_epn > 0:
            edge_ratio = p95_epn / mean_epn
            # Only widen if p95 density exceeds mean (ratio > 1)
            if edge_ratio > 1.0:
                effective_bpn = int(bytes_per_node * edge_ratio)

    # DERIVED: mem_budget = effective_free × margin / effective_bytes_per_node
    mem_budget = int(effective_free * _SAFETY_MARGIN / effective_bpn)

    # --- Step 6: cg_ratio + throughput floor ---
    # cg_ratio uses training-adjusted β (forward β × backward_multiplier)
    # to reflect actual training regime, not forward-only inference.
    cg_ratio = None
    throughput_floor = None
    bwd_mult = backward_multiplier if backward_multiplier is not None else _GRAD_MULTIPLIER

    if gamma is not None and beta is not None:
        beta_train = beta * bwd_mult
        gamma_per_node = gamma / max(1.0, mean_nodes)
        gamma_eff = gamma_per_node / max(1, num_workers)
        if beta_train > 0:
            cg_ratio = gamma_eff / beta_train

        # Throughput floor: minimum batch size (in nodes) to amortize GPU
        # kernel overhead α. Below this, GPU wastes cycles on per-step
        # overhead. Above this, throughput is flat (collation-bound) or
        # monotonically increasing (compute-bound).
        #
        # Derivation (from audit §1.5):
        #   B_floor = α / (γ/W − β_train·m̄)   [graphs, exists when γ/W > β_train·m̄]
        #   N_floor = B_floor · m̄              [nodes, for DynamicBatchSampler]
        alpha_train = alpha * bwd_mult if alpha is not None else 0.0
        collation_rate = gamma / max(1, num_workers)  # sec/graph with W workers
        gpu_rate = beta_train * mean_nodes             # sec/graph on GPU
        if collation_rate > gpu_rate and alpha_train > 0:
            # Collation-dominated regime: throughput floor exists
            throughput_floor_graphs = alpha_train / (collation_rate - gpu_rate)
            throughput_floor = max(1, int(math.ceil(
                throughput_floor_graphs * mean_nodes
            )))

    # --- Step 7: budget = clamp(throughput_floor, mem_ceiling) ---
    if gamma is None:
        budget = max(1, mem_budget)
        binding = "fallback"
    elif throughput_floor is not None and throughput_floor > mem_budget:
        # Floor exceeds ceiling — GPU overhead can't be amortized within
        # VRAM. Use mem_budget (best we can do), warn user.
        budget = max(1, mem_budget)
        binding = "memory"
        log.warning("throughput_floor_exceeds_vram",
                    floor=throughput_floor, ceiling=mem_budget)
    elif throughput_floor is not None and throughput_floor > 1:
        budget = max(throughput_floor, mem_budget)
        binding = "memory"  # floor < ceiling, ceiling is binding
    else:
        budget = max(1, mem_budget)
        binding = "memory"

    log.info("node_budget",
             budget=budget, mem_budget=mem_budget,
             throughput_floor=throughput_floor, binding=binding,
             cg_ratio=round(cg_ratio, 2) if cg_ratio is not None else None,
             free_vram_gb=round(effective_free / 1e9, 2),
             bytes_per_node=bytes_per_node,
             effective_bpn=effective_bpn,
             backward_multiplier=(round(backward_multiplier, 2)
                                  if backward_multiplier is not None else None),
             mean_nodes=round(mean_nodes, 1))

    return BudgetResult(
        budget=budget, mean_nodes=mean_nodes,
        mem_budget=mem_budget, throughput_floor=throughput_floor,
        binding=binding, cg_ratio=cg_ratio,
        bytes_per_node=bytes_per_node,
        gamma_us=round(gamma * 1e6, 1) if gamma is not None else None,
        alpha_ms=round(alpha * 1000, 2) if alpha is not None else None,
        beta_us=round(beta * 1e6, 3) if beta is not None else None,
        edges_per_node_p95=edges_per_node_p95,
        backward_multiplier=backward_multiplier,
        teacher_vram_bytes=teacher_vram,
        is_compiled=is_compiled,
    )
