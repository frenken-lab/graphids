"""VRAM budget → batch size → worker count.

Wraps ``torch.profiler`` to measure forward/backward memory and timing.
Peak VRAM is decomposed into:
- ``fixed_overhead`` — teacher params (frozen, on-GPU during forward only
  but batch-size-invariant). Reserved from ``free`` before sizing.
- ``bytes_per_node`` / ``bytes_per_edge`` — scaling cost. Budget picks the
  max nodes / edges that fit in ``(free - fixed) × safety``.

Workers sized by ``ceil((t_io + t_collation) / t_gpu)``.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass

import torch
import torch.profiler as tp
from torch_geometric.data import Batch

from graphids._otel import get_logger
from graphids.config.topology import cache_dir

log = get_logger(__name__)


VALID_CONV_TYPES = frozenset({"gat", "gatv2", "transformer", "gps"})


def collect_batch(dataset, target_nodes: int) -> Batch:
    """Collect graphs until reaching ``target_nodes`` total. No DataLoader overhead."""
    graphs, total = [], 0
    for g in dataset:
        graphs.append(g)
        total += g.num_nodes
        if total >= target_nodes:
            break
    return Batch.from_data_list(graphs)


def _find_teacher(model) -> torch.nn.Module | None:
    """Locate the frozen KD teacher.

    ``_attach_teacher`` (distillation.py) stashes the teacher in
    ``module.__dict__['teacher']`` on the loss module — so the access path
    is ``model.loss_fn.teacher``. Falls back to ``model.teacher`` for any
    legacy wiring.
    """
    loss = getattr(model, "loss_fn", None)
    t = getattr(loss, "teacher", None) if loss is not None else None
    return t if t is not None else getattr(model, "teacher", None)


def _teacher_param_bytes(model) -> int:
    t = _find_teacher(model)
    if t is None:
        return 0
    return sum(p.numel() * p.element_size() for p in t.parameters())


def _settings():
    from graphids.config.settings import get_settings
    s = get_settings()
    return s.budget_safety_margin, s.budget_grad_mult, s.budget_fallback_bpn

_SAFETY, _GRAD_MULT, _FALLBACK_BPN = _settings()


@dataclass
class BudgetResult:
    """Output of ``BudgetProfiler.node_budget``."""
    budget: int                       # max nodes per batch
    mean_nodes: float
    binding: str
    bytes_per_node: int | None = None
    bytes_per_edge: int | None = None  # per-edge scaling cost (from same probe)
    edge_budget: int | None = None     # max edges per batch (dual constraint)
    fixed_overhead: int = 0            # teacher params + framework (non-scaling)
    backward_multiplier: float | None = None
    t_fwd: float = 0.0
    t_io: float = 0.0                  # median dataset __getitem__ time / sample


class BudgetProfiler:
    """Profile VRAM usage and compute batch/worker sizing.

    Inherits measurement from ``torch.profiler.profile`` with
    ``record_function`` tags to separate forward from backward peaks.
    """

    @staticmethod
    def probe(model, batch) -> tuple[int, int, float, float, int]:
        """Profile one forward + one training step via ``torch.profiler``.

        Peak VRAM is split into ``fixed`` (teacher params — on-GPU during
        forward but batch-invariant) and ``scalable`` (activations +
        gradients, which grow with V and E). Per-node and per-edge bpn are
        both reported; the sampler uses whichever is more restrictive.

        Returns:
            bytes_per_node: ``scalable_peak / num_nodes``
            bytes_per_edge: ``scalable_peak / num_edges``
            backward_multiplier: ``bwd_peak / fwd_peak`` (measured)
            t_fwd_seconds: forward CUDA time in seconds
            fixed_bytes: teacher param bytes (reserved before sizing)
        """
        dev = model.device
        was_training = model.training
        model.eval()
        fn = getattr(model, "_step", None) or model

        # Warmup: JIT, autotuning, AND CUDA profiler (CUPTI) init.
        # Wrapping in tp.profile absorbs profiler setup cost so the
        # measurement pass below doesn't include it.  See issue #28.
        with tp.profile(activities=[tp.ProfilerActivity.CUDA], profile_memory=True):
            with torch.no_grad():
                fn(batch)
        torch.cuda.synchronize(dev)

        # Forward-only pass: measure fwd peak + timing
        with tp.profile(activities=[tp.ProfilerActivity.CUDA], profile_memory=True) as fwd_p:
            with torch.profiler.record_function("forward"):
                with torch.no_grad():
                    fn(batch)
        torch.cuda.synchronize(dev)
        fwd_evts = fwd_p.key_averages()
        fwd_peak = max((e.cuda_memory_usage for e in fwd_evts), default=0)
        t_fwd = sum(e.cuda_time_total for e in fwd_evts) / 1e6

        # Full training step: measure bwd peak
        bwd_peak = fwd_peak
        step_fn = getattr(model, "_step", None)
        if step_fn is not None:
            model.train()
            with tp.profile(activities=[tp.ProfilerActivity.CUDA], profile_memory=True) as bwd_p:
                with torch.profiler.record_function("train_step"):
                    loss = step_fn(batch.clone())
                    if isinstance(loss, (tuple, list)):
                        loss = loss[0]
                    elif isinstance(loss, dict):
                        loss = loss["loss"]
                with torch.profiler.record_function("backward"):
                    loss.backward()
            torch.cuda.synchronize(dev)
            bwd_peak = max((e.cuda_memory_usage for e in bwd_p.key_averages()), default=fwd_peak)
            model.zero_grad(set_to_none=True)

        model.train() if was_training else model.eval()

        # Decompose: teacher params are fixed overhead. Only activations/grads
        # scale with batch V/E, so the per-node / per-edge cost should be
        # computed over the *scalable* peak, not the full peak.
        fixed = _teacher_param_bytes(model)
        scalable = max(1, bwd_peak - fixed)
        bpn_node = max(1, scalable // max(1, batch.num_nodes))
        bpn_edge = max(1, scalable // max(1, int(batch.num_edges)))
        bwd_mult = max(1.0, bwd_peak / fwd_peak) if fwd_peak > 0 else _GRAD_MULT
        return bpn_node, bpn_edge, bwd_mult, t_fwd, fixed

    @staticmethod
    def node_budget(
        dataset: str, lake_root: str, *, conv_type: str = "gatv2",
        heads: int = 4, model=None, train_dataset=None,
    ) -> BudgetResult:
        """Pack budget: ``(free - fixed) × safety / bpn`` per dimension.

        Returns both node and edge budgets from the same probe. The sampler
        packs graphs honoring whichever constraint binds first per batch.
        """
        if conv_type not in VALID_CONV_TYPES:
            raise ValueError(
                f"Unknown conv_type {conv_type!r}; expected one of {sorted(VALID_CONV_TYPES)}"
            )

        stats = json.loads((cache_dir(lake_root, dataset) / "cache_metadata.json").read_text())["graph_stats"]
        mean_nodes = stats["node_count"]["mean"]
        free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 12 * 1024**3

        # gps conv: hardcoded formula pending proper profiling (see budget TODO).
        # Skip probe since gps attention memory scales as O((V+E)^2) and the
        # linear-probe model doesn't fit.
        if conv_type == "gps":
            b = int(math.sqrt(free / (heads * 6)))
            return BudgetResult(budget=b, mean_nodes=mean_nodes, binding="memory")

        bpn_node, bpn_edge = _FALLBACK_BPN, 0
        bwd, t_fwd, fixed = None, 0.0, 0
        if model and train_dataset and torch.cuda.is_available():
            b = collect_batch(train_dataset, 2000).to(model.device)
            bpn_node, bpn_edge, bwd, t_fwd, fixed = BudgetProfiler.probe(model, b)
            del b; torch.cuda.empty_cache()

        # Reserve fixed overhead (teacher params) before scaling math. Teacher
        # is on-GPU during every forward; its bytes don't grow with V/E so
        # they belong on the ``free`` side of the budget, not ``bpn``.
        free_scalable = max(1, free - fixed)

        # NOTE: an edge-density ``r`` correction used to inflate ``bpn_node``
        # against the dataset p95/mean edge ratio. It was defensive padding
        # because the single-axis node budget couldn't see edge-heavy
        # batches coming. With ``bpn_edge`` now probed separately and the
        # sampler enforcing a dual constraint, the r-inflation double-pads
        # and wastes node capacity — dropped.

        budget = max(1, int(free_scalable * _SAFETY / max(1, bpn_node)))
        edge_budget = max(1, int(free_scalable * _SAFETY / bpn_edge)) if bpn_edge > 0 else None
        binding = "memory" if (model and train_dataset) else "fallback"
        return BudgetResult(
            budget=budget, mean_nodes=mean_nodes, binding=binding,
            bytes_per_node=bpn_node, bytes_per_edge=bpn_edge,
            edge_budget=edge_budget, fixed_overhead=fixed,
            backward_multiplier=bwd, t_fwd=t_fwd,
        )

    @staticmethod
    def autosize_workers(
        model, dataset, result: BudgetResult, *, default_prefetch: int = 2,
    ) -> tuple[int, int]:
        """``ceil((t_io + t_collation) / t_gpu)`` → ``(num_workers, prefetch_factor)``.

        Worker time has two components: dataset ``__getitem__`` (real I/O —
        dominates on NFS-resident data) + ``Batch.from_data_list`` (collation
        — CPU-bound, small). The earlier implementation measured only
        collation and undersized workers whenever data lived off-TMPDIR.
        """
        from graphids._slurm import slurm_cpus_per_task
        if model.device.type != "cuda" or result.t_fwd <= 0:
            return 2, default_prefetch

        t_gpu = result.t_fwd * (result.backward_multiplier or _GRAD_MULT)
        batch = collect_batch(dataset, result.budget)
        if batch.num_graphs < 2:
            return 2, default_prefetch

        # Drain pending CUDA work so async ops don't inflate CPU timing (#28)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)

        # I/O timing: sample ``batch.num_graphs`` indices and walk dataset
        # ``__getitem__``. This is the path a real worker takes per batch.
        n = min(batch.num_graphs, len(dataset))
        idx = random.sample(range(len(dataset)), n)
        ti = []
        for _ in range(3):
            t0 = time.perf_counter()
            for i in idx:
                _ = dataset[i]
            ti.append(time.perf_counter() - t0)
        t_io = statistics.median(ti)

        # Collation timing (operates on already-loaded Data objects)
        graphs = batch.to_data_list()
        tc = []
        for _ in range(3):
            t0 = time.perf_counter(); Batch.from_data_list(graphs); tc.append(time.perf_counter() - t0)
        t_coll = statistics.median(tc)

        t_worker = t_io + t_coll
        result.t_io = t_io

        max_cpus = slurm_cpus_per_task() or os.cpu_count()
        w = max(1, min(math.ceil(t_worker / t_gpu), max(1, max_cpus - 2)))
        return w, 4 if w >= 8 else 2


# Convenience aliases for callers that don't want to type BudgetProfiler.
node_budget = BudgetProfiler.node_budget
autosize_workers = BudgetProfiler.autosize_workers
