"""VRAM budget → batch size → worker count.

Wraps ``torch.profiler`` to measure forward/backward memory and timing,
then applies:
- budget = free_vram × safety / bytes_per_node
- workers = ceil(t_collation / t_gpu)
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass

import torch
import torch.profiler as tp
from torch_geometric.data import Batch

from graphids._otel import get_logger
from graphids.config.topology import cache_dir
from graphids.core.data.sampler import collect_batch

log = get_logger(__name__)
VALID_CONV_TYPES = frozenset({"gat", "gatv2", "transformer", "gps"})


def _settings():
    from graphids.config.settings import get_settings
    s = get_settings()
    return s.budget_safety_margin, s.budget_grad_mult, s.budget_fallback_bpn

_SAFETY, _GRAD_MULT, _FALLBACK_BPN = _settings()


@dataclass
class BudgetResult:
    """Output of ``BudgetProfiler.node_budget``."""
    budget: int
    mean_nodes: float
    binding: str
    bytes_per_node: int | None = None
    backward_multiplier: float | None = None
    t_fwd: float = 0.0


class BudgetProfiler:
    """Profile VRAM usage and compute batch/worker sizing.

    Inherits measurement from ``torch.profiler.profile`` with
    ``record_function`` tags to separate forward from backward peaks.
    """

    @staticmethod
    def probe(model, batch) -> tuple[int, float, float]:
        """Profile one forward + one training step via ``torch.profiler``.

        Uses ``record_function("forward")`` and ``record_function("backward")``
        to tag phases within a single profiler pass.

        Returns:
            bytes_per_node: peak VRAM (including backward) / num_nodes
            backward_multiplier: bwd_peak / fwd_peak (measured, not assumed)
            t_fwd_seconds: forward CUDA time in seconds
        """
        dev = model.device
        was_training = model.training
        model.eval()
        fn = getattr(model, "_step", None) or model

        # Warmup (JIT, autotuning)
        with torch.no_grad():
            fn(batch)
        torch.cuda.synchronize(dev)

        # Forward-only pass: measure fwd peak + timing
        with tp.profile(activities=[tp.ProfilerActivity.CUDA], profile_memory=True) as fwd_p:
            with torch.profiler.record_function("forward"):
                with torch.no_grad():
                    fn(batch)
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
            bwd_peak = max((e.cuda_memory_usage for e in bwd_p.key_averages()), default=fwd_peak)
            model.zero_grad(set_to_none=True)

        model.train() if was_training else model.eval()
        bpn = max(1, bwd_peak // max(1, batch.num_nodes))
        bwd_mult = max(1.0, bwd_peak / fwd_peak) if fwd_peak > 0 else _GRAD_MULT
        return bpn, bwd_mult, t_fwd

    @staticmethod
    def node_budget(
        dataset: str, lake_root: str, *, conv_type: str = "gatv2",
        heads: int = 4, model=None, train_dataset=None,
    ) -> BudgetResult:
        """``free_vram × safety / bytes_per_node`` → max nodes per batch."""
        stats = json.loads((cache_dir(lake_root, dataset) / "cache_metadata.json").read_text())["graph_stats"]
        mean_nodes = stats["node_count"]["mean"]
        free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 12 * 1024**3
        if model and getattr(model, "teacher", None):
            free -= int(sum(p.numel() * p.element_size() for p in model.teacher.parameters()) * 2.5)

        if conv_type == "gps":
            return BudgetResult(int(math.sqrt(free / (heads * 6))), mean_nodes, "memory")

        bpn, bwd, t_fwd = _FALLBACK_BPN, None, 0.0
        if model and train_dataset and torch.cuda.is_available():
            b = collect_batch(train_dataset, 2000).to(model.device)
            bpn, bwd, t_fwd = BudgetProfiler.probe(model, b)
            del b; torch.cuda.empty_cache()

        es, ns = stats.get("edge_count"), stats["node_count"]
        if es and "p95" in es and "p95" in ns:
            r = (es["p95"] / max(1.0, ns["p95"])) / max(1e-9, es["mean"] / max(1.0, ns["mean"]))
            if r > 1.0:
                bpn = int(bpn * r)

        budget = max(1, int(free * _SAFETY / bpn))
        binding = "memory" if (model and train_dataset) else "fallback"
        return BudgetResult(budget, mean_nodes, binding, bpn, bwd, t_fwd)

    @staticmethod
    def autosize_workers(
        model, dataset, result: BudgetResult, *, default_prefetch: int = 2,
    ) -> tuple[int, int]:
        """``ceil(t_collation / t_gpu)`` → ``(num_workers, prefetch_factor)``."""
        from graphids._slurm import slurm_cpus_per_task
        if model.device.type != "cuda" or result.t_fwd <= 0:
            return 2, default_prefetch

        t_gpu = result.t_fwd * (result.backward_multiplier or _GRAD_MULT)
        batch = collect_batch(dataset, result.budget)
        if batch.num_graphs < 2:
            return 2, default_prefetch

        graphs = batch.to_data_list()
        tc = []
        for _ in range(3):
            t0 = time.perf_counter(); Batch.from_data_list(graphs); tc.append(time.perf_counter() - t0)

        max_cpus = slurm_cpus_per_task() or os.cpu_count()
        w = max(1, min(math.ceil(statistics.median(tc) / t_gpu), max(1, max_cpus - 2)))
        return w, 4 if w >= 8 else 2


# Convenience aliases for callers that don't want to type BudgetProfiler.
node_budget = BudgetProfiler.node_budget
autosize_workers = BudgetProfiler.autosize_workers
