"""Empirical CUDA budget probing."""

from __future__ import annotations

import math

import torch
from structlog import get_logger
from torch_geometric.data import Batch

from .config import MB, BudgetConfig
from .diagnostics import _loss, _silent_log
from .measure import _measure_fwd, _measure_fwd_bwd
from .stats import _dataset_size_tensors
from .types import BudgetResult

log = get_logger(__name__)


def probe(
    model,
    train_dataset,
    *,
    quadratic: bool = False,
    min_steps: int | None = None,
    config: BudgetConfig | None = None,
) -> BudgetResult:
    """Estimate a packing budget from actual training graphs."""
    if not torch.cuda.is_available() or model is None or train_dataset is None:
        raise RuntimeError("budget probe requires CUDA + model + train_dataset")

    from graphids.core.data.datamodule.sampler import pack_offline

    cfg = config or BudgetConfig.from_env()
    dev = model.device
    rng_devices = [dev.index] if dev.index is not None else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(cfg.probe_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.probe_seed)
        return _probe_body(
            model,
            train_dataset,
            dev,
            pack_offline,
            quadratic=quadratic,
            min_steps=min_steps,
            config=cfg,
        )


def _probe_body(
    model,
    train_dataset,
    dev,
    pack_offline,
    *,
    quadratic: bool,
    min_steps: int | None,
    config: BudgetConfig,
) -> BudgetResult:
    step_fn = getattr(model, "_step", None) or (lambda b: model.training_step(b, 0))
    was_training = model.training

    sizes_t, edges_t = _dataset_size_tensors(train_dataset, config=config)
    sizes_list = [int(v) for v in sizes_t.tolist()]
    if not sizes_list:
        raise RuntimeError("budget probe: train_dataset is empty")

    b0_nodes = int(sizes_t.max().item())
    b0_edges = int(edges_t.max().item())
    plans_0 = pack_offline(sizes_t, max_num=b0_nodes, edge_sizes=edges_t, max_edges=b0_edges)
    if not plans_0:
        raise RuntimeError("pack_offline returned 0 plans under B0 budget")

    plan_v = torch.tensor([int(sizes_t[plan].sum().item()) for plan in plans_0])
    plan_e = torch.tensor([int(edges_t[plan].sum().item()) for plan in plans_0])
    candidate_indices = sorted({int(plan_v.argmax()), int(plan_e.argmax())})

    model.train()
    fwd_peak = 0
    fwd_time = 0.0
    candidate_peaks: list[tuple[int, int, int]] = []

    with torch.enable_grad(), _silent_log(model):
        warm_batch = Batch.from_data_list(
            [train_dataset[i] for i in plans_0[candidate_indices[0]]]
        ).to(dev)
        for _ in range(3):
            _loss(step_fn(warm_batch)).backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        baseline = torch.cuda.memory_allocated(dev)

        model.eval()
        fwd_peak, fwd_time = _measure_fwd(model, step_fn, warm_batch, dev)
        model.train()
        del warm_batch
        torch.cuda.empty_cache()

        for ci in candidate_indices:
            cb = Batch.from_data_list([train_dataset[i] for i in plans_0[ci]]).to(dev)
            c_v, c_e = int(cb.num_nodes), int(cb.num_edges)
            c_peak = _measure_fwd_bwd(model, step_fn, cb, dev, debug_tag=f"candidate_ci{ci}")
            candidate_peaks.append((c_v, c_e, c_peak))
            del cb
            torch.cuda.empty_cache()

    worst_v, worst_e, worst_peak = max(candidate_peaks, key=lambda p: p[2])

    free = torch.cuda.mem_get_info()[0]
    target = max(1, int(free * config.safety_margin))
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    optim_overhead = 2 * param_bytes
    cudnn_reserve = int(free * config.cudnn_reserve)
    headroom = max(1, target - baseline - optim_overhead - cudnn_reserve)

    activation = max(1, worst_peak - baseline)
    bwd_mult = max(1.0, worst_peak / max(1, fwd_peak))

    if quadratic:
        alpha = activation / (worst_v * worst_v)
        b1 = max(b0_nodes, int(math.sqrt(headroom / alpha)))
    else:
        per_node = activation / worst_v
        b1 = max(b0_nodes, int(headroom / per_node))

    if min_steps is not None and min_steps > 1:
        total_nodes = sum(sizes_list)
        step_cap = total_nodes // min_steps
        if step_cap > b0_nodes:
            b1 = min(b1, step_cap)

    epn = worst_e / max(1, worst_v)
    edge_budget = max(b0_edges, int(b1 * epn * config.edge_headroom))

    repack_done = False
    sanity_v = sanity_peak = 0
    if b1 > b0_nodes:
        repack_done = True
        plans_1 = pack_offline(sizes_t, max_num=b1, edge_sizes=edges_t, max_edges=edge_budget)
        if not plans_1:
            raise RuntimeError("pack_offline returned 0 plans under B1 budget")
        plan1_v = torch.tensor([int(sizes_t[plan].sum().item()) for plan in plans_1])
        sci = int(plan1_v.argmax())
        sb = Batch.from_data_list([train_dataset[i] for i in plans_1[sci]]).to(dev)
        sanity_v = int(sb.num_nodes)
        with torch.enable_grad(), _silent_log(model):
            sanity_peak = _measure_fwd_bwd(model, step_fn, sb, dev, debug_tag="sanity")
        del sb
        torch.cuda.empty_cache()
        if sanity_peak > target:
            model.train(was_training)
            raise RuntimeError(
                f"budget probe: post-repack sanity probe exceeded target. "
                f"V={sanity_v} peak={sanity_peak // MB}MB "
                f"target={target // MB}MB free={free // MB}MB. "
                f"Lower max budget (e.g. set GRAPHIDS_BUDGET_SAFETY_MARGIN"
                f"<{config.safety_margin}), reduce window/dataset, or use a larger GPU."
            )

    model.train(was_training)

    log.info(
        "budget_probed",
        quadratic=quadratic,
        free_mb=free // MB,
        baseline_mb=baseline // MB,
        worst_V=worst_v,
        worst_E=worst_e,
        worst_peak_mb=worst_peak // MB,
        activation_mb=activation // MB,
        optim_overhead_mb=optim_overhead // MB,
        cudnn_reserve_mb=cudnn_reserve // MB,
        target_mb=target // MB,
        budget_nodes=b1,
        budget_edges=edge_budget,
        edges_per_node=round(epn, 2),
        bwd_mult=round(bwd_mult, 2),
        t_fwd_ms=round(fwd_time * 1000, 1),
        repacked=repack_done,
        sanity_V=sanity_v,
        sanity_peak_mb=sanity_peak // MB,
        n_candidates=len(candidate_peaks),
    )
    return BudgetResult(
        budget=b1,
        edge_budget=edge_budget,
        backward_multiplier=bwd_mult,
        t_fwd=float(fwd_time),
        target_bytes=target,
    )
