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

GPS uses a **three-point quadratic probe** instead (``probe_quadratic``):
global attention memory scales as O(V²·heads), so a linear fit is wrong.
The quadratic coefficients (α, β, γ) are solved for via least-squares,
and the node budget is the positive root of ``α·V² + β·V + γ = free·safety``.
Edge budget is derived from the empirical edges-per-node ratio on the
probe batches themselves — not a hardcoded multiplier.

Workers sized by ``ceil((t_io + t_collation) / t_gpu)``.
"""

from __future__ import annotations

import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Batch

from graphids._otel import get_logger

log = get_logger(__name__)


# Three disjoint states for how BudgetResult.budget was derived. Named by
# *why* (the reason for this value), not *what* (its implementation). Stamped
# as MLflow tag ``graphids.budget_binding`` by
# MLflowTrainingCallback._check_budget_utilization — downstream dashboards
# filter on these exact strings, so rename impact is cross-cutting.
BudgetBinding = Literal[
    "measured",  # probe ran, fit was valid
    "measured_degenerate_fallback",  # probe ran but fit degenerate → formula
    "opted_in_fallback",  # prereqs missing + GRAPHIDS_ALLOW_FALLBACK_BUDGET=1
]


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
    binding: BudgetBinding = "measured"
    backward_multiplier: float | None = None
    t_fwd: float = 0.0
    # Bytes the budget was solved against (= free * safety at probe time).
    # Consumed by MLflowTrainingCallback to warn on under-utilization.
    target_bytes: int = 0


_FALLBACK_ENV = "GRAPHIDS_ALLOW_FALLBACK_BUDGET"


def _fallback_allowed() -> bool:
    return os.environ.get(_FALLBACK_ENV, "0") == "1"


def _fallback_budget_pair(free: int, heads: int) -> tuple[int, int]:
    """Conservative (node_budget, edge_budget) pair when a real probe can't run.

    Called by (a) _gps_budget's opt-in-fallback branch and (b) node_budget's
    linear opt-in-fallback branch. For the GPS degenerate-fit branch, only
    the node half of this pair is used — the edge half there comes from the
    empirical edges-per-node measured on the (failed) probe batches.
    """
    from graphids.config.settings import get_settings

    s = get_settings()
    budget = max(1, int(math.sqrt(free / (heads * s.gps_fallback_attention_divisor))))
    edge_budget = int(budget * s.fallback_edge_node_ratio)
    return budget, edge_budget


def _require_probe_prereqs(model, train_dataset, conv_type: str) -> None:
    """Raise unless CUDA + model + train_dataset are all present, or the env
    opt-in is set. Silent fallbacks produce conservative budgets that can
    leave 60-80% of GPU memory on the table — a correctness hazard that
    looks like a successful run. Fail loudly instead.
    """
    if _fallback_allowed():
        return
    missing = []
    if not torch.cuda.is_available():
        missing.append("CUDA")
    if model is None:
        missing.append("model")
    if train_dataset is None:
        missing.append("train_dataset")
    if missing:
        raise RuntimeError(
            f"budget probe prerequisites missing: {', '.join(missing)} "
            f"(conv_type={conv_type}). Set {_FALLBACK_ENV}=1 to allow a "
            f"conservative hardcoded budget (may silently under-utilize GPU)."
        )


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


def probe_quadratic(model, batches: list[Batch]) -> tuple[float, float, float, float]:
    """Three-point quadratic fit of ``peak_bwd = α·V² + β·V + γ`` for GPS.

    Runs warmup on the smallest batch (cuDNN autotuning, allocator baseline),
    then full fwd+bwd on every batch and collects (V, peak) pairs. Fits the
    quadratic via least-squares in float64 — the intercept γ absorbs every
    fixed cost, α isolates the O(V²) attention term, β picks up O(V) linear
    contributions (MPNN branch + feature projections).

    Returns ``(alpha, beta, gamma, t_fwd_last)``. Caller solves the quadratic
    against the free-VRAM target and handles degenerate fits.

    Caller owns batch lifecycles; batches should be ordered ascending by
    ``num_nodes`` so the smallest is used for warmup.
    """
    dev = model.device

    with _eval_mode(model):
        with torch.no_grad():
            (getattr(model, "_step", None) or model)(batches[0])
    torch.cuda.synchronize(dev)

    vs: list[int] = []
    peaks: list[int] = []
    t_fwd_last = 0.0
    for b in batches:
        _, bwd_peak, t_fwd = _step_peaks(model, b)
        vs.append(int(b.num_nodes))
        peaks.append(bwd_peak)
        t_fwd_last = t_fwd

    # polyfit returns coefficients in ASCENDING order: [gamma, beta, alpha].
    gamma, beta, alpha = np.polynomial.polynomial.polyfit(vs, peaks, 2)
    return float(alpha), float(beta), float(gamma), t_fwd_last


_GPS_PROBE_SIZES = (500, 1500, 4000)


def _gps_budget(dataset: str, free: int, heads: int, model, train_dataset) -> BudgetResult:
    """Quadratic-probe path for GPS. Returns a ``BudgetResult`` with both
    node and edge budgets. Raises if probe prerequisites (CUDA + model +
    dataset) are missing unless ``GRAPHIDS_ALLOW_FALLBACK_BUDGET=1``. Falls
    back silently only when the quadratic fit itself degenerates (α≤0 or
    negative discriminant) — that's a measurement edge case, not a missing
    prerequisite.
    """
    _require_probe_prereqs(model, train_dataset, conv_type="gps")

    if not torch.cuda.is_available() or model is None or train_dataset is None:
        # Opted-in fallback (env var set). Conservative sqrt formula with a
        # node-to-edge ratio (from settings) that will never be the binding axis.
        budget, edge_budget = _fallback_budget_pair(free, heads)
        target = max(1, int(free * _SAFETY))
        return BudgetResult(
            budget=budget,
            edge_budget=edge_budget,
            binding="opted_in_fallback",
            target_bytes=target,
        )

    dev = model.device
    batches = [collect_batch(train_dataset, n).clone().to(dev) for n in _GPS_PROBE_SIZES]
    total_v = sum(int(b.num_nodes) for b in batches)
    total_e = sum(int(b.num_edges) for b in batches)
    epn = total_e / max(1, total_v)

    alpha, beta, gamma, t_fwd = probe_quadratic(model, batches)
    del batches
    torch.cuda.empty_cache()

    from graphids.config.settings import get_settings

    settings = get_settings()
    target = free * _SAFETY
    roots = np.roots([alpha, beta, gamma - target])
    real_positive = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    if not real_positive:
        # Degenerate fit — node half of the fallback pair is the right conservative
        # sqrt budget; edge_budget below uses the empirical epn from the probe
        # batches (not the fallback's 10× ratio) since we actually measured it.
        budget, _ = _fallback_budget_pair(free, heads)
        binding = "measured_degenerate_fallback"
        log.warning(
            "gps_probe_degenerate",
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            roots=roots.tolist(),
        )
    else:
        budget = max(1, int(max(real_positive)))
        binding = "measured"

    edge_budget = max(1, int(budget * epn * settings.empirical_epn_headroom))

    log.info(
        "budget_probed",
        dataset=dataset,
        conv_type="gps",
        binding=binding,
        free_mb=free // (1024 * 1024),
        alpha_bytes_per_v2=round(alpha, 4),
        beta_bytes_per_v=round(beta, 2),
        gamma_bytes=int(gamma),
        budget_nodes=budget,
        budget_edges=edge_budget,
        edges_per_node=round(epn, 2),
        t_fwd_ms=round(t_fwd * 1000, 1),
    )
    return BudgetResult(
        budget=budget,
        edge_budget=edge_budget,
        binding=binding,
        t_fwd=t_fwd,
        target_bytes=int(target),
    )


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

    if conv_type == "gps":
        return _gps_budget(dataset, free, heads, model, train_dataset)

    _require_probe_prereqs(model, train_dataset, conv_type=conv_type)

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

    from graphids.config.settings import get_settings

    settings = get_settings()
    free_scalable = max(1, int(free * _SAFETY))
    budget = max(1, free_scalable // max(1, bpn_node))
    # bpn_edge=0 only reachable via opt-in fallback; give a non-None edge
    # budget so pack_offline's dual-budget invariant holds. fallback_edge_node_ratio
    # (default 10×) is a conservative ceiling — real CAN graphs sit at 1–10 edges/node.
    edge_budget = (
        max(1, free_scalable // bpn_edge)
        if bpn_edge > 0
        else int(budget * settings.fallback_edge_node_ratio)
    )
    binding = "measured" if (model and train_dataset) else "opted_in_fallback"
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
        target_bytes=free_scalable,
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
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    max_cpus = (int(slurm_cpus) if slurm_cpus and slurm_cpus.isdigit() else None) or os.cpu_count()
    w = max(1, min(math.ceil(t_worker / t_gpu), max(1, max_cpus - 2)))
    return w, 4 if w >= 8 else 2
