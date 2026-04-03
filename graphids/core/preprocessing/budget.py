"""Node budget for PyG's DynamicBatchSampler.

    budget = min(memory_ceiling, throughput_ceiling)

Memory ceiling: max nodes that fit in VRAM.
Throughput ceiling: batch size where CPU collation keeps up with GPU.
    Only exists when collation is slower AND GPU has per-step overhead.

Cost model: docs/reference/gnn_throughput_equations.md
Each constant is tagged DERIVED / HEURISTIC / FALLBACK.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import structlog
import torch
from torch.utils.benchmark import Timer as BenchmarkTimer

from graphids.config import cache_dir

log = structlog.get_logger()

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
    throughput_budget: int | None  # pipeline ceiling in nodes, None if N/A
    binding: str                 # "memory" | "throughput" | "fallback"
    cg_ratio: float | None       # (γ/W)/β diagnostic — >1 = collation bottleneck
    # Raw probe measurements (for post-hoc analysis)
    bytes_per_node: int | None = None
    gamma_us: float | None = None   # γ: collation cost per graph (μs)
    alpha_ms: float | None = None   # α: GPU overhead per step (ms)
    beta_us: float | None = None    # β: GPU cost per node (μs)
    # Edge-aware fields (Phase 1B)
    bytes_per_edge: int | None = None         # B: per-edge VRAM cost from 2×2 probe
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


def _probe(model, dataset, step_fn=None) -> tuple[int, int, float, float, float, float]:
    """Measure per-node/edge VRAM, backward multiplier, γ, α, β.

    Edge-aware: measures VRAM at two batch sizes with different N/E counts,
    solves the 2×2 system:  vram = A·N + B·E  for per-node (A) and per-edge (B).

    Backward: runs one forward+backward step to measure the real gradient
    memory multiplier instead of using _GRAD_MULTIPLIER heuristic.

    Timing: fits  T_gpu(N) = α + β·N  from two-point measurement.

    Returns (bytes_per_node, bytes_per_edge, backward_multiplier, gamma, alpha, beta).
    bytes_per_node and bytes_per_edge include the backward multiplier.
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

    # --- γ: collation rate (CPU, single sample) ---
    # Deterministic CPU work, single measurement is representative.
    t0 = time.perf_counter()
    batch_large = Batch.from_data_list(graphs_large)
    t_collate = time.perf_counter() - t0
    gamma = t_collate / len(graphs_large)  # seconds per graph

    batch_small = Batch.from_data_list(graphs_small)
    batch_large = batch_large.to(model.device)
    batch_small = batch_small.to(model.device)
    nodes_large = batch_large.num_nodes
    nodes_small = batch_small.num_nodes
    edges_large = batch_large.num_edges
    edges_small = batch_small.num_edges

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

    # --- Forward-only VRAM at two batch sizes ---
    # Solve 2×2 system: vram = A·N + B·E for per-node (A) and per-edge (B) cost.
    if model.device.type == "cuda":
        # Small batch
        torch.cuda.reset_peak_memory_stats(model.device)
        before = torch.cuda.memory_allocated(model.device)
        with torch.no_grad():
            fn(batch_small)
        torch.cuda.synchronize()
        vram_small = torch.cuda.max_memory_allocated(model.device) - before

        # Large batch
        torch.cuda.reset_peak_memory_stats(model.device)
        before = torch.cuda.memory_allocated(model.device)
        with torch.no_grad():
            fn(batch_large)
        torch.cuda.synchronize()
        vram_large = torch.cuda.max_memory_allocated(model.device) - before
    else:
        vram_small, vram_large = 0, 0

    # Solve: vram = A·N + B·E
    #   A = (vram_s·E_l - vram_l·E_s) / (N_s·E_l - N_l·E_s)
    #   B = (N_s·vram_l - N_l·vram_s) / (N_s·E_l - N_l·E_s)
    det = nodes_small * edges_large - nodes_large * edges_small
    if abs(det) > 0 and vram_small > 0 and vram_large > 0:
        fwd_per_node = max(1, int((vram_small * edges_large - vram_large * edges_small) / det))
        fwd_per_edge = max(0, int((nodes_small * vram_large - nodes_large * vram_small) / det))
    else:
        # Degenerate case (same E/N ratio) — fall back to node-only
        fwd_per_node = max(1, int(vram_large / max(1, nodes_large))) if vram_large > 0 else 1
        fwd_per_edge = 0

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
        except Exception:
            log.warning("backward_probe_failed", fallback=_GRAD_MULTIPLIER)
            model.eval()

    # Apply backward multiplier to get training-time costs
    bytes_per_node = int(fwd_per_node * backward_multiplier)
    bytes_per_edge = int(fwd_per_edge * backward_multiplier)

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
             bytes_per_node=bytes_per_node, bytes_per_edge=bytes_per_edge,
             backward_multiplier=round(backward_multiplier, 2),
             fwd_per_node=fwd_per_node, fwd_per_edge=fwd_per_edge,
             nodes_small=nodes_small, nodes_large=nodes_large,
             edges_small=edges_small, edges_large=edges_large,
             n_graphs=len(graphs_large),
             t_collate_ms=round(t_collate * 1000, 1),
             t_gpu_small_ms=round(t_gpu_small * 1000, 1),
             t_gpu_large_ms=round(t_gpu_large * 1000, 1),
             alpha_ms=round(alpha * 1000, 2),
             beta_us=round(beta * 1e6, 3),
             gamma_us=round(gamma * 1e6, 1))

    return bytes_per_node, bytes_per_edge, backward_multiplier, gamma, alpha, beta


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
    2. Read free VRAM.
    3. GPS conv → quadratic VRAM formula, return early.
    4. If model available → _probe() for bytes_per_node, γ, α, β.
    5. mem_budget = free × SAFETY_MARGIN / bytes_per_node.
    6. throughput_budget = α / (γ/W - β·m̄) × m̄  [if it exists].
    7. budget = min(mem_budget, throughput_budget).
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
            mem_budget=budget, throughput_budget=None,
            binding="memory", cg_ratio=None,
            teacher_vram_bytes=teacher_vram, is_compiled=is_compiled,
        )

    # --- Step 4: probe ---
    gamma = alpha = beta = None
    bytes_per_node = _FALLBACK_BYTES_PER_NODE
    bytes_per_edge = 0
    backward_multiplier = None

    if model is not None and train_dataset is not None and torch.cuda.is_available():
        step_fn = getattr(model, "_step", None)
        bytes_per_node, bytes_per_edge, backward_multiplier, gamma, alpha, beta = (
            _probe(model, train_dataset, step_fn=step_fn)
        )

    # --- Step 5: memory ceiling ---
    # Edge-aware: effective cost per node accounts for edge density (Phase 1B).
    # Uses p95 edge/node ratio for conservative sizing.
    edges_per_node_p95 = None
    if bytes_per_edge > 0 and edge_stats is not None:
        edges_per_node_p95 = edge_stats["p95"] / node_stats["p95"]
        effective_bpn = bytes_per_node + bytes_per_edge * edges_per_node_p95
    else:
        effective_bpn = bytes_per_node

    # DERIVED: mem_budget = effective_free × margin / effective_bytes_per_node
    mem_budget = int(effective_free * _SAFETY_MARGIN / effective_bpn)

    # --- Step 6: throughput ceiling ---
    # DERIVED from setting T_collate/W = T_gpu and solving for batch size B:
    #
    #   γ·B / W = α + β·B·m̄
    #   B·(γ/W - β·m̄) = α
    #   B = α / (γ/W - β·m̄)           ← in graphs
    #   N = B · m̄                      ← convert to nodes
    #
    # Exists only when:
    #   gap = γ/W - β·m̄ > 0   (collation slower than GPU per graph)
    #   α > 0                  (measurable per-step overhead)
    throughput_budget = None
    cg_ratio = None

    if gamma is not None:
        # Diagnostic ratio: (γ/W) / β — not used for decisions, just logged.
        gamma_per_node = gamma / max(1.0, mean_nodes)
        gamma_eff = gamma_per_node / max(1, num_workers)
        if beta > 0:
            cg_ratio = gamma_eff / beta

        # Throughput budget from the balance equation.
        delivery_rate = gamma / max(1, num_workers)   # γ/W
        beta_per_graph = beta * mean_nodes             # β·m̄
        gap = delivery_rate - beta_per_graph

        if gap > 0 and alpha > 0:
            optimal_graphs = max(1, int(alpha / gap))
            throughput_budget = int(optimal_graphs * mean_nodes)

    # --- Step 7: final budget = min(memory, throughput) ---
    if throughput_budget is not None and throughput_budget < mem_budget:
        budget = throughput_budget
        binding = "throughput"
    else:
        budget = mem_budget
        binding = "fallback" if gamma is None else "memory"

    budget = max(1, budget)

    log.info("node_budget",
             budget=budget, mem_budget=mem_budget,
             throughput_budget=throughput_budget, binding=binding,
             cg_ratio=round(cg_ratio, 2) if cg_ratio is not None else None,
             num_workers=num_workers,
             free_vram_gb=round(effective_free / 1e9, 2),
             bytes_per_node=bytes_per_node,
             bytes_per_edge=bytes_per_edge,
             edges_per_node_p95=(round(edges_per_node_p95, 2)
                                 if edges_per_node_p95 is not None else None),
             backward_multiplier=(round(backward_multiplier, 2)
                                  if backward_multiplier is not None else None),
             teacher_vram_mb=round(teacher_vram / 1e6, 1) if teacher_vram else None,
             mean_nodes=round(mean_nodes, 1),
             alpha_ms=round(alpha * 1000, 2) if alpha is not None else None,
             beta_us=round(beta * 1e6, 3) if beta is not None else None)

    return BudgetResult(
        budget=budget, mean_nodes=mean_nodes,
        mem_budget=mem_budget, throughput_budget=throughput_budget,
        binding=binding, cg_ratio=cg_ratio,
        bytes_per_node=bytes_per_node,
        gamma_us=round(gamma * 1e6, 1) if gamma is not None else None,
        alpha_ms=round(alpha * 1000, 2) if alpha is not None else None,
        beta_us=round(beta * 1e6, 3) if beta is not None else None,
        bytes_per_edge=bytes_per_edge if bytes_per_edge else None,
        edges_per_node_p95=edges_per_node_p95,
        backward_multiplier=backward_multiplier,
        teacher_vram_bytes=teacher_vram,
        is_compiled=is_compiled,
    )
