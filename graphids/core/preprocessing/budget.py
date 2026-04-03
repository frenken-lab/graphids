"""Node budget for PyG's DynamicBatchSampler.

Memory ceiling: max nodes that fit in VRAM (hard upper bound — OOM above).
Worker sizing: calibrate_at_budget() measures real T_collation and T_gpu
at the operating batch size → compute_resource_profile() derives workers.

Cost model: docs/reference/gnn_throughput_equations.md
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
_GRAD_MULTIPLIER = 2

# FALLBACK: conservative bytes/node when no model is available for probing.
# Real models use 1-8KB/node. Overestimating is safe (smaller budget, no OOM).
_FALLBACK_BYTES_PER_NODE = 32_768

# DERIVED: conv types with O(N²) global attention (not O(E) over edges).
_QUADRATIC_CONV_TYPES = frozenset({"gps"})


def _collect_graphs(dataset, target_nodes: int) -> list:
    """Collect graphs from dataset until reaching target_nodes total."""
    graphs, total = [], 0
    for g in dataset:
        graphs.append(g)
        total += g.num_nodes
        if total >= target_nodes:
            break
    return graphs


@dataclass
class BudgetResult:
    """Output of node_budget()."""
    budget: int                  # max_num for DynamicBatchSampler
    mean_nodes: float            # dataset mean from cache_metadata.json
    mem_budget: int              # VRAM ceiling in nodes
    binding: str                 # "memory" | "fallback"
    bytes_per_node: int | None = None
    edges_per_node_p95: float | None = None
    backward_multiplier: float | None = None
    teacher_vram_bytes: int = 0
    is_compiled: bool | None = None


@dataclass
class ResourceProfile:
    """Full resource sizing from the GPU-first sizing chain.

    Computed by compute_resource_profile() from calibrate_at_budget() data.
    See docs/backlog/training-efficiency.md for the sizing chain derivation.
    """
    node_budget: int
    graphs_per_batch: int
    t_collation_us: float      # measured at operating batch size
    t_gpu_us: float            # measured forward × backward_multiplier
    workers: int               # ceil(t_collation / t_gpu), capped to max_cpus
    prefetch_factor: int       # 2 for ≤4 workers, 4 for ≥8
    cpus: int                  # workers + 2
    memory_gb: int             # workers × rss + base + headroom


# ---------------------------------------------------------------------------
# Calibration: measure at the real operating batch size
# ---------------------------------------------------------------------------

def calibrate_at_budget(
    model, dataset, budget: int,
    *, backward_multiplier: float = _GRAD_MULTIPLIER,
) -> tuple[float, float]:
    """Measure T_collation and T_gpu at the actual operating batch size.

    Builds one batch targeting ``budget`` nodes from the dataset, then:
    - T_collation: wall-clock CPU time for Batch.from_data_list (3-sample median)
    - T_gpu: forward-only GPU time (BenchmarkTimer) × backward_multiplier

    No extrapolation — direct measurement at the operating point.

    Returns (t_collation_s, t_gpu_s) in seconds, or (0, 0) on failure.
    """
    from torch_geometric.data import Batch

    if model.device.type != "cuda":
        return 0.0, 0.0

    graphs = _collect_graphs(dataset, budget)
    if len(graphs) < 2:
        return 0.0, 0.0

    # --- T_collation: 3-sample median ---
    torch.cuda.synchronize()
    gc.collect()

    collation_samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        batch = Batch.from_data_list(graphs)
        collation_samples.append(time.perf_counter() - t0)
    t_collation = statistics.median(collation_samples)

    # --- T_gpu: forward at budget size × backward_multiplier ---
    from graphids.core.models._training import eval_mode

    batch = batch.to(model.device)
    fn = getattr(model, "_step", None) or model

    with eval_mode(model):
        # Warmup (JIT, autotuning)
        with torch.no_grad():
            fn(batch)
        torch.cuda.synchronize()

        def _fwd():
            with torch.no_grad():
                fn(batch)

        t_gpu_fwd = BenchmarkTimer(
            stmt="_fwd()", globals={"_fwd": _fwd},
        ).blocked_autorange(min_run_time=0.2).median

    t_gpu = t_gpu_fwd * backward_multiplier

    del batch
    torch.cuda.empty_cache()

    log.info("calibrate_at_budget",
             n_graphs=len(graphs),
             t_collation_ms=round(t_collation * 1000, 1),
             t_gpu_fwd_ms=round(t_gpu_fwd * 1000, 1),
             t_gpu_ms=round(t_gpu * 1000, 1),
             backward_multiplier=round(backward_multiplier, 2))

    return t_collation, t_gpu


def compute_resource_profile(
    result: BudgetResult,
    *,
    t_collation_s: float,
    t_gpu_s: float,
    max_cpus: int | None = None,
    worker_rss_gb: float = 1.5,
    base_rss_gb: float = 4.0,
) -> ResourceProfile | None:
    """Derive workers/prefetch/CPUs/memory from calibration measurements.

    t_collation_s and t_gpu_s come from calibrate_at_budget().
    Returns None when t_gpu_s <= 0.
    max_cpus caps workers to fit SLURM allocation (read SLURM_CPUS_PER_TASK).
    """
    if t_gpu_s <= 0:
        return None

    t_collation_us = t_collation_s * 1e6
    t_gpu_us = t_gpu_s * 1e6
    graphs_per_batch = max(1, int(result.budget / max(1.0, result.mean_nodes)))

    workers = max(1, math.ceil(t_collation_us / t_gpu_us))

    # Cap to available CPUs (leave 2 for main process + headroom)
    if max_cpus is not None:
        workers = min(workers, max(1, max_cpus - 2))

    prefetch_factor = 4 if workers >= 8 else 2
    cpus = workers + 2
    memory_gb = max(16, math.ceil(workers * worker_rss_gb + base_rss_gb + 4))

    log.info("resource_profile",
             workers=workers, prefetch_factor=prefetch_factor,
             cpus=cpus, memory_gb=memory_gb,
             t_collation_us=round(t_collation_us, 1),
             t_gpu_us=round(t_gpu_us, 1),
             graphs_per_batch=graphs_per_batch,
             cg_ratio=round(t_collation_us / t_gpu_us, 2))

    return ResourceProfile(
        node_budget=result.budget,
        graphs_per_batch=graphs_per_batch,
        t_collation_us=round(t_collation_us, 1),
        t_gpu_us=round(t_gpu_us, 1),
        workers=workers,
        prefetch_factor=prefetch_factor,
        cpus=cpus,
        memory_gb=memory_gb,
    )


# ---------------------------------------------------------------------------
# VRAM probe: bytes_per_node + backward multiplier
# ---------------------------------------------------------------------------

def _extract_loss(output):
    """Handle _step() return formats: scalar, tuple, or dict."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        return output["loss"]
    raise TypeError(f"Cannot extract loss from {type(output)}")


def _probe_vram(model, dataset, step_fn=None) -> tuple[int, float]:
    """Measure bytes_per_node and backward_multiplier on a small batch.

    Collects ~2000 nodes, runs one forward pass for VRAM measurement,
    one forward+backward pass for backward multiplier.

    Returns (bytes_per_node, backward_multiplier).
    bytes_per_node includes the backward multiplier.
    """
    from torch_geometric.data import Batch

    # ~2000 nodes — large enough for stable VRAM measurement,
    # small enough to be fast during DataLoader setup.
    from graphids.core.models._training import eval_mode

    graphs = _collect_graphs(dataset, 2000)
    batch = Batch.from_data_list(graphs).to(model.device)
    fn = step_fn or model

    with eval_mode(model):
        # Warmup: torch.compile, kernel JIT, cuDNN autotuning
        with torch.no_grad():
            fn(batch)
        if model.device.type == "cuda":
            torch.cuda.synchronize()

        # --- Forward-only VRAM ---
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
            before = torch.cuda.memory_allocated(model.device)
            with torch.no_grad():
                fn(batch)
            torch.cuda.synchronize()
            vram = torch.cuda.max_memory_allocated(model.device) - before
        else:
            vram = 0

        fwd_per_node = max(1, int(vram / max(1, batch.num_nodes))) if vram > 0 else 1

        # --- Backward multiplier ---
        # Temporarily switch to train mode for fwd+bwd VRAM measurement
        backward_multiplier = _GRAD_MULTIPLIER
        if model.device.type == "cuda" and step_fn is not None:
            try:
                model.train()
                batch_bwd = batch.clone()
                torch.cuda.reset_peak_memory_stats(model.device)
                before = torch.cuda.memory_allocated(model.device)
                loss = _extract_loss(step_fn(batch_bwd))
                torch.autograd.backward(loss)
                torch.cuda.synchronize()
                bwd_peak = torch.cuda.max_memory_allocated(model.device) - before
                if vram > 0:
                    backward_multiplier = max(1.0, bwd_peak / vram)
                model.zero_grad(set_to_none=True)
                del batch_bwd, loss
                model.eval()
            except Exception as exc:
                log.warning("backward_probe_failed", fallback=_GRAD_MULTIPLIER,
                            error=str(exc)[:120])
                model.eval()

    bytes_per_node = int(fwd_per_node * backward_multiplier)

    del batch
    if model.device.type == "cuda":
        torch.cuda.empty_cache()

    log.info("vram_probe",
             bytes_per_node=bytes_per_node,
             backward_multiplier=round(backward_multiplier, 2),
             fwd_per_node=fwd_per_node, num_nodes=total_nodes)

    return bytes_per_node, backward_multiplier


# ---------------------------------------------------------------------------
# Node budget: VRAM ceiling → max nodes per batch
# ---------------------------------------------------------------------------

def node_budget(
    dataset: str,
    lake_root: str,
    *,
    conv_type: str = "gatv2",
    heads: int = 4,
    model=None,
    train_dataset=None,
) -> BudgetResult:
    """Compute max_num for DynamicBatchSampler(mode="node").

    1. Read mean_nodes from cache_metadata.json.
    2. Read free VRAM (after model load, before optimizer).
    3. GPS conv → quadratic VRAM formula, return early.
    4. If model available → _probe_vram() for bytes_per_node + backward_multiplier.
    5. Apply edge-aware margin.
    6. budget = free × SAFETY_MARGIN / effective_bytes_per_node.
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

    teacher_vram = 0
    if model is not None and hasattr(model, "teacher") and model.teacher is not None:
        teacher_params = sum(p.numel() * p.element_size() for p in model.teacher.parameters())
        teacher_vram = int(teacher_params * 2.5)
        log.info("kd_teacher_vram", bytes=teacher_vram, mb=round(teacher_vram / 1e6, 1))
    effective_free = free - teacher_vram

    is_compiled = hasattr(model, "_orig_mod") if model is not None else None

    # --- Step 3: quadratic conv types ---
    if conv_type in _QUADRATIC_CONV_TYPES:
        budget = int(math.sqrt(effective_free / (heads * 3 * 2)))
        log.info("node_budget", conv_type=conv_type, budget=budget,
                 free_vram_gb=round(effective_free / 1e9, 2), binding="memory")
        return BudgetResult(
            budget=budget, mean_nodes=mean_nodes, mem_budget=budget,
            binding="memory",
            teacher_vram_bytes=teacher_vram, is_compiled=is_compiled,
        )

    # --- Step 4: VRAM probe ---
    bytes_per_node = _FALLBACK_BYTES_PER_NODE
    backward_multiplier = None
    probed = False

    if model is not None and train_dataset is not None and torch.cuda.is_available():
        step_fn = getattr(model, "_step", None)
        bytes_per_node, backward_multiplier = (
            _probe_vram(model, train_dataset, step_fn=step_fn)
        )
        probed = True

    # --- Step 5: edge-aware margin ---
    edges_per_node_p95 = None
    effective_bpn = bytes_per_node
    if edge_stats is not None and "p95" in edge_stats and "p95" in node_stats:
        mean_epn = edge_stats["mean"] / max(1.0, node_stats["mean"])
        p95_epn = edge_stats["p95"] / max(1.0, node_stats["p95"])
        edges_per_node_p95 = p95_epn
        if mean_epn > 0:
            edge_ratio = p95_epn / mean_epn
            if edge_ratio > 1.0:
                effective_bpn = int(bytes_per_node * edge_ratio)

    # --- Step 6: budget = VRAM ceiling ---
    mem_budget = int(effective_free * _SAFETY_MARGIN / effective_bpn)
    budget = max(1, mem_budget)
    binding = "memory" if probed else "fallback"

    log.info("node_budget",
             budget=budget, mem_budget=mem_budget, binding=binding,
             free_vram_gb=round(effective_free / 1e9, 2),
             bytes_per_node=bytes_per_node, effective_bpn=effective_bpn,
             backward_multiplier=(round(backward_multiplier, 2)
                                  if backward_multiplier is not None else None),
             mean_nodes=round(mean_nodes, 1))

    return BudgetResult(
        budget=budget, mean_nodes=mean_nodes, mem_budget=mem_budget,
        binding=binding,
        bytes_per_node=bytes_per_node,
        edges_per_node_p95=edges_per_node_p95,
        backward_multiplier=backward_multiplier,
        teacher_vram_bytes=teacher_vram,
        is_compiled=is_compiled,
    )
