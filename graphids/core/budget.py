"""Budget probe v4: one measurement + allocator baseline.

The slope-fit was extracting fixed cost as the y-intercept of two probe
points. The allocator already knows that number — ``torch.cuda.memory_allocated``
after warmup IS the fixed cost (params + gradient buffers + optimizer state
+ cuDNN workspace cache). Subtract it and divide. No polyfit, no roots.

```
peak       = baseline + activation(V)
activation = peak - torch.cuda.memory_allocated()       (one measurement)
per_node   = activation / V        (linear)
per_v2     = activation / V²       (quadratic, GPS)
budget     = solve(target = baseline + activation(V_budget))
```

Composes:
- ``torch_geometric.profile.profileit`` → peak VRAM + step time
- ``torch.cuda.memory_allocated`` → baseline (free, no probe)
- ``torch.cuda.mem_get_info`` → free VRAM
- ``torch.utils.benchmark.Timer`` → CPU timing for autosize_workers
- ``os.sched_getaffinity`` → CPU cap

Public surface matches ``budget.py``: ``BudgetResult``, ``node_budget``,
``autosize_workers``, ``collect_batch``.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass

import torch
from structlog import get_logger
from torch.utils.benchmark import Timer
from torch_geometric.data import Batch
from torch_geometric.profile import profileit

log = get_logger(__name__)

_SAFETY = float(os.environ.get("GRAPHIDS_BUDGET_SAFETY_MARGIN", "0.95"))
_EPN_HEADROOM = float(os.environ.get("GRAPHIDS_EMPIRICAL_EPN_HEADROOM", "1.1"))
_MB = 1024 * 1024


@dataclass(frozen=True)
class BudgetResult:
    budget: int
    edge_budget: int
    binding: str = "measured"
    backward_multiplier: float = 2.0
    t_fwd: float = 0.0
    target_bytes: int = 0


def collect_batch(dataset, target_nodes: int) -> Batch:
    graphs, total = [], 0
    for g in dataset:
        graphs.append(g)
        total += g.num_nodes
        if total >= target_nodes:
            break
    return Batch.from_data_list(graphs)


def _loss(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict):
        return out["loss"]
    return out


@contextlib.contextmanager
def _silent_log(model):
    """Silence ``self.log(...)`` calls during the probe so warmup + profileit
    backward passes don't pollute MLflow with non-training metric points.
    Restores the original method on exit. Probe never calls ``optimizer.step()``,
    so model weights are unchanged after probe even though grads are computed.
    """
    sentinel = object()
    orig = model.__dict__.get("log", sentinel)
    model.log = lambda *a, **k: None
    try:
        yield
    finally:
        if orig is sentinel:
            del model.log
        else:
            model.log = orig


def probe(
    model,
    train_dataset,
    *,
    probe_nodes: int = 10000,
    quadratic: bool = False,
) -> BudgetResult:
    """One probe step. Baseline-subtract. Solve.

    Quadratic mode is for GPS-style global attention (memory ∝ V²·heads).
    """
    if not torch.cuda.is_available() or model is None or train_dataset is None:
        raise RuntimeError("budget probe requires CUDA + model + train_dataset")

    dev = model.device
    # `forward()` returns architecture outputs (logits / latents / tuples), not
    # a scalar loss — backward fails. Use `_step` if defined; otherwise route
    # through `training_step` which returns a scalar loss for every model.
    step_fn = getattr(model, "_step", None) or (lambda b: model.training_step(b, 0))
    batch = collect_batch(train_dataset, probe_nodes).clone().to(dev)
    V, E = int(batch.num_nodes), int(batch.num_edges)

    was_training = model.training
    model.train()
    with _silent_log(model):
        for _ in range(3):
            _loss(step_fn(batch)).backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)

        baseline = torch.cuda.memory_allocated(dev)

        @profileit(device=dev.index if dev.index is not None else 0)
        def _fwd():
            with torch.no_grad():
                _loss(step_fn(batch))

        @profileit(device=dev.index if dev.index is not None else 0)
        def _fwd_bwd():
            _loss(step_fn(batch)).backward()
            model.zero_grad(set_to_none=True)

        model.eval()
        _, fwd_stats = _fwd()
        model.train()
        _, bwd_stats = _fwd_bwd()
    model.train(was_training)

    peak = int(bwd_stats.max_allocated_gpu * _MB)
    fwd_peak = int(fwd_stats.max_allocated_gpu * _MB)
    activation = max(1, peak - baseline)

    free = torch.cuda.mem_get_info()[0]
    target = max(1, int(free * _SAFETY))
    headroom = max(1, target - baseline)

    if quadratic:
        alpha = activation / (V * V)
        budget = max(1, int(math.sqrt(headroom / alpha)))
    else:
        per_node = activation / V
        budget = max(1, int(headroom / per_node))

    epn = E / max(1, V)
    edge_budget = max(1, int(budget * epn * _EPN_HEADROOM))
    bwd_mult = max(1.0, peak / max(1, fwd_peak))

    del batch
    torch.cuda.empty_cache()

    log.info(
        "budget_probed",
        quadratic=quadratic,
        free_mb=free // _MB,
        baseline_mb=baseline // _MB,
        peak_mb=peak // _MB,
        activation_mb=activation // _MB,
        budget_nodes=budget,
        budget_edges=edge_budget,
        edges_per_node=round(epn, 2),
        bwd_mult=round(bwd_mult, 2),
        t_fwd_ms=round(fwd_stats.time * 1000, 1),
    )
    return BudgetResult(
        budget=budget,
        edge_budget=edge_budget,
        backward_multiplier=bwd_mult,
        t_fwd=float(fwd_stats.time),
        target_bytes=target,
    )


def node_budget(
    dataset: str,
    *,
    model=None,
    train_dataset=None,
    conv_type: str | None = None,
    heads: int | None = None,
) -> BudgetResult:
    if conv_type is None and model is not None:
        conv_type = getattr(model.hparams, "conv_type", "gatv2")
    return probe(model, train_dataset, quadratic=(conv_type == "gps"))


def _cpu_cap() -> int:
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        n = int(slurm)
    else:
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:
            n = os.cpu_count() or 4
    return max(1, n - 2)


def autosize_workers(
    model,
    dataset,
    result: BudgetResult,
    *,
    default_prefetch: int = 2,
) -> tuple[int, int, dict]:
    if model is None or model.device.type != "cuda" or result.t_fwd <= 0:
        return 2, default_prefetch, {}
    batch = collect_batch(dataset, result.budget)
    if batch.num_graphs < 2:
        return 2, default_prefetch, {}
    torch.cuda.synchronize(model.device)

    n = min(batch.num_graphs, len(dataset))
    t_io = Timer(
        stmt="[ds[i] for i in range(n)]",
        globals={"ds": dataset, "n": n},
    ).blocked_autorange(min_run_time=0.1).median
    graphs = batch.to_data_list()
    t_coll = Timer(
        stmt="Batch.from_data_list(graphs)",
        globals={"Batch": Batch, "graphs": graphs},
    ).blocked_autorange(min_run_time=0.1).median

    t_gpu = result.t_fwd * result.backward_multiplier
    raw = math.ceil((t_io + t_coll) / t_gpu)
    cap = _cpu_cap()
    w = max(1, min(raw, cap))
    pf = 4 if w >= 8 else default_prefetch
    diag = {
        "t_io_ms": round(t_io * 1000, 2),
        "t_coll_ms": round(t_coll * 1000, 2),
        "t_gpu_ms": round(t_gpu * 1000, 2),
        "ratio": round((t_io + t_coll) / t_gpu, 3),
        "raw_w": raw,
        "cap": cap,
    }
    log.info(
        "workers_autosized",
        nw=w,
        prefetch_factor=pf,
        source="capped" if raw > cap else "measured",
        **diag,
    )
    return w, pf, diag
