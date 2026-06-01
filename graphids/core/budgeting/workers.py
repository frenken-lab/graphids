"""DataLoader worker autosizing helpers."""

from __future__ import annotations

import math
import os

import torch
from structlog import get_logger
from torch.utils.benchmark import Timer
from torch_geometric.data import Batch

from .types import BudgetResult

log = get_logger(__name__)


def collect_batch(dataset, target_nodes: int) -> Batch:
    graphs, total = [], 0
    for g in dataset:
        graphs.append(g)
        total += g.num_nodes
        if total >= target_nodes:
            break
    return Batch.from_data_list(graphs)


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
    t_io = (
        Timer(
            stmt="[ds[i] for i in range(n)]",
            globals={"ds": dataset, "n": n},
        )
        .blocked_autorange(min_run_time=0.1)
        .median
    )
    graphs = batch.to_data_list()
    t_coll = (
        Timer(
            stmt="Batch.from_data_list(graphs)",
            globals={"Batch": Batch, "graphs": graphs},
        )
        .blocked_autorange(min_run_time=0.1)
        .median
    )

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
