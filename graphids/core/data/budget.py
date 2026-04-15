"""VRAM budget → batch size → worker count.

Measures forward/backward memory via the CUDA allocator high-water mark
(``torch.cuda.max_memory_allocated``) and timing via wall-clock around
``torch.cuda.synchronize``. The probe is **two-point**: runs forward +
backward on a small batch and a larger one, then takes the slope
``(peak_big - peak_small) / (nodes_big - nodes_small)`` as ``bpn_node``.
The y-intercept of that line absorbs every fixed cost (model params,
optimizer state, cuDNN workspaces, allocator baseline, KD teacher) —
no resident-subtract heuristic. Single-point estimates systematically
over-charge small batches with the fixed costs, inflating bpn_node by
~3-4× and capping packed batches well below the real hardware limit.

Workers sized by ``ceil((t_io + t_collation) / t_gpu)``.
"""

from __future__ import annotations

import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch_geometric.data import Batch

from graphids._otel import get_logger

log = get_logger(__name__)


_BWD_MULT_FALLBACK = 2.0


def _settings() -> tuple[float, int]:
    from graphids.config.settings import get_settings

    s = get_settings()
    return s.budget_safety_margin, s.budget_fallback_bpn


_SAFETY, _FALLBACK_BPN = _settings()


@dataclass(frozen=True)
class BudgetResult:
    """Output of ``node_budget`` — the sampler's sizing contract."""

    budget: int  # max nodes per batch
    edge_budget: int | None = None
    binding: str = "memory"
    backward_multiplier: float | None = None
    t_fwd: float = 0.0


@contextmanager
def _eval_mode(model):
    """Save/restore ``model.training`` around a probe. See critical-constraints.md."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def collect_batch(dataset, target_nodes: int) -> Batch:
    """Collect graphs until reaching ``target_nodes`` total. No DataLoader overhead."""
    graphs, total = [], 0
    for g in dataset:
        graphs.append(g)
        total += g.num_nodes
        if total >= target_nodes:
            break
    return Batch.from_data_list(graphs)


def _step_peaks(model, batch) -> tuple[int, int, float]:
    """Measure (fwd_peak, bwd_peak, t_fwd) for one batch on a warmed model.

    Caller owns warmup + lifecycle.
    """
    dev = model.device
    fn = getattr(model, "_step", None) or model

    with _eval_mode(model):
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            fn(batch)
        torch.cuda.synchronize(dev)
        t_fwd = time.perf_counter() - t0
        fwd_peak = torch.cuda.max_memory_allocated(dev)

    bwd_peak = fwd_peak
    step_fn = getattr(model, "_step", None)
    if step_fn is not None:
        model.train()
        torch.cuda.reset_peak_memory_stats(dev)
        loss = step_fn(batch)
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        elif isinstance(loss, dict):
            loss = loss["loss"]
        loss.backward()
        torch.cuda.synchronize(dev)
        bwd_peak = torch.cuda.max_memory_allocated(dev)
        model.zero_grad(set_to_none=True)

    return fwd_peak, bwd_peak, t_fwd


def probe(model, batch_small, batch_big) -> tuple[int, int, float, float]:
    """Two-point linear fit of VRAM vs. batch size.

    Runs a warmup on ``batch_small`` (trigger lazy CUDA init, cuDNN
    autotuning, kernel JIT, allocator baseline), then full fwd+bwd passes
    on both batches. ``bpn_node`` is the slope of ``bwd_peak`` vs. nodes
    across the two points; the intercept (fixed overhead) drops out.
    Same for ``bpn_edge``.

    Returns ``(bpn_node, bpn_edge, bwd_mult, t_fwd_seconds)``. ``t_fwd``
    is from the larger batch, which is more representative of real
    training steps than the warmup-sized one.

    Caller owns batch lifecycles.
    """
    dev = model.device

    # Warmup on the smaller batch — sets up cuDNN autotuning + allocator cache.
    with _eval_mode(model):
        with torch.no_grad():
            (getattr(model, "_step", None) or model)(batch_small)
    torch.cuda.synchronize(dev)

    fwd_s, bwd_s, _ = _step_peaks(model, batch_small)
    fwd_b, bwd_b, t_fwd_big = _step_peaks(model, batch_big)

    dn = max(1, batch_big.num_nodes - batch_small.num_nodes)
    de = max(1, int(batch_big.num_edges) - int(batch_small.num_edges))
    # Clamp to avoid pathologies when allocator caching makes the big probe
    # report lower peak than the small one (rare but possible at small deltas).
    bpn_node = max(1, (bwd_b - bwd_s) // dn)
    bpn_edge = max(1, (bwd_b - bwd_s) // de)

    fwd_scaling = max(1, fwd_b - fwd_s)
    bwd_scaling = max(1, bwd_b - bwd_s)
    bwd_mult = max(1.0, bwd_scaling / fwd_scaling) if fwd_scaling > 0 else _BWD_MULT_FALLBACK
    return bpn_node, bpn_edge, bwd_mult, t_fwd_big


def node_budget(
    dataset: str,
    *,
    conv_type: str = "gatv2",
    heads: int = 4,
    model=None,
    train_dataset=None,
) -> BudgetResult:
    """Pack budget: ``free × safety / bpn`` per dimension.

    ``free`` from ``mem_get_info`` already excludes resident allocation, and
    ``bpn`` from the probe is purely batch-scaling — so one multiply gives
    the max batch that fits without double-counting.
    """
    free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 12 * 1024**3

    # gps conv: hardcoded formula pending proper profiling. Skip probe since
    # gps attention memory scales as O((V+E)^2) and the linear probe can't fit.
    if conv_type == "gps":
        b = int(math.sqrt(free / (heads * 6)))
        return BudgetResult(budget=b, binding="memory")

    bpn_node, bpn_edge = _FALLBACK_BPN, 0
    bwd, t_fwd = None, 0.0
    if model and train_dataset and torch.cuda.is_available():
        dev = model.device
        # Two-point probe. Small batch amortizes warmup / cuDNN autotuning;
        # big batch drives the slope. 10× ratio is enough for the fixed-cost
        # intercept to cancel out without inflating probe runtime.
        small = collect_batch(train_dataset, 2000).clone().to(dev)
        big = collect_batch(train_dataset, 20000).clone().to(dev)
        bpn_node, bpn_edge, bwd, t_fwd = probe(model, small, big)
        del small, big
        torch.cuda.empty_cache()

    free_scalable = max(1, int(free * _SAFETY))
    budget = max(1, free_scalable // max(1, bpn_node))
    edge_budget = max(1, free_scalable // bpn_edge) if bpn_edge > 0 else None
    binding = "memory" if (model and train_dataset) else "fallback"
    log.info(
        "budget_probed",
        dataset=dataset,
        conv_type=conv_type,
        binding=binding,
        free_mb=free // (1024 * 1024),
        bpn_node=bpn_node,
        bpn_edge=bpn_edge,
        budget_nodes=budget,
        budget_edges=edge_budget,
        bwd_mult=round(bwd, 2) if bwd is not None else None,
        t_fwd_ms=round(t_fwd * 1000, 1),
    )
    return BudgetResult(
        budget=budget,
        edge_budget=edge_budget,
        binding=binding,
        backward_multiplier=bwd,
        t_fwd=t_fwd,
    )


def autosize_workers(
    model,
    dataset,
    result: BudgetResult,
    *,
    default_prefetch: int = 2,
) -> tuple[int, int]:
    """``ceil((t_io + t_collation) / t_gpu)`` → ``(num_workers, prefetch_factor)``.

    Worker time has two components: dataset ``__getitem__`` (real I/O) plus
    ``Batch.from_data_list`` (collation, CPU-bound).
    """
    from graphids._slurm import slurm_cpus_per_task

    if model is None or model.device.type != "cuda" or result.t_fwd <= 0:
        return 2, default_prefetch

    t_gpu = result.t_fwd * (result.backward_multiplier or _BWD_MULT_FALLBACK)
    batch = collect_batch(dataset, result.budget)
    if batch.num_graphs < 2:
        return 2, default_prefetch

    # Drain pending CUDA work so async ops don't inflate CPU timing (#28)
    torch.cuda.synchronize(model.device)

    # I/O timing: sample batch.num_graphs indices, walk dataset __getitem__.
    n = min(batch.num_graphs, len(dataset))
    idx = random.sample(range(len(dataset)), n)
    t0 = time.perf_counter()
    for i in idx:
        _ = dataset[i]
    t_io = time.perf_counter() - t0

    # Collation timing (operates on already-loaded Data objects)
    graphs = batch.to_data_list()
    t0 = time.perf_counter()
    Batch.from_data_list(graphs)
    t_coll = time.perf_counter() - t0

    t_worker = t_io + t_coll
    max_cpus = slurm_cpus_per_task() or os.cpu_count()
    w = max(1, min(math.ceil(t_worker / t_gpu), max(1, max_cpus - 2)))
    return w, 4 if w >= 8 else 2
