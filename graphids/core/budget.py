"""GPU/CPU budget probe helpers."""

from __future__ import annotations

import contextlib
import math
import os
import time
from dataclasses import dataclass

import torch
from structlog import get_logger
from torch.utils.benchmark import Timer
from torch_geometric.data import Batch

log = get_logger(__name__)

_SAFETY = float(os.environ.get("GRAPHIDS_BUDGET_SAFETY_MARGIN", "0.85"))
_EPN_HEADROOM = float(os.environ.get("GRAPHIDS_EMPIRICAL_EPN_HEADROOM", "1.1"))
# Fixed RNG seed for the probe. Combined with torch.random.fork_rng(), this
# makes probe forwards bit-deterministic (same draw → same NaN/no-NaN every
# run) AND isolates probe RNG consumption from subsequent training. Override
# via env var when bisecting a flaky NaN to find a non-failing seed.
_PROBE_SEED = int(os.environ.get("GRAPHIDS_PROBE_SEED", "20260506"))
# Reserved fraction of free VRAM for cuDNN multi-shape workspace cache. The
# probe sees one (V,E) shape; pack_offline produces many distinct shapes per
# epoch, each potentially triggering a cuDNN benchmark allocation. 5% covers
# typical growth on V100/H100. Tighten if log shows headroom unused.
_CUDNN_RESERVE = float(os.environ.get("GRAPHIDS_BUDGET_CUDNN_RESERVE", "0.05"))
_HEURISTIC_BPN = int(os.environ.get("GRAPHIDS_BUDGET_BYTES_PER_NODE", str(256 * 1024)))
_HEURISTIC_BPE = int(os.environ.get("GRAPHIDS_BUDGET_BYTES_PER_EDGE", str(32 * 1024)))
_HEURISTIC_GPS_BPN2 = float(os.environ.get("GRAPHIDS_BUDGET_GPS_BYTES_PER_NODE2", "32768"))
_DEFAULT_TARGET_BYTES = int(
    os.environ.get("GRAPHIDS_BUDGET_DEFAULT_TARGET_BYTES", str(8 * 1024**3))
)
_DEFAULT_EDGES_PER_NODE = float(os.environ.get("GRAPHIDS_BUDGET_EDGES_PER_NODE", "4.0"))
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
    """Silence ``self.log(...)`` during probe warmup and measurement."""
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


def _dump_intermediates(
    model, batch, tag: str, *, cpu_state=None, cuda_state=None, dev=None
) -> None:
    """Replay a failing forward under saved RNG state and log tensor finiteness."""
    diag: dict = {"tag": tag, "V": int(batch.num_nodes), "E": int(batch.num_edges)}
    bad_params = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad_params.append(name)
    diag["bad_params"] = bad_params

    if hasattr(model, "_forward_tensors"):
        if cpu_state is not None:
            torch.set_rng_state(cpu_state)
        if cuda_state is not None and dev is not None:
            torch.cuda.set_rng_state(cuda_state, dev)
        with torch.no_grad():
            ea = getattr(batch, "edge_attr", None)
            out = model._forward_tensors(
                batch.x, batch.edge_index, batch.batch, edge_attr=ea, node_id=batch.node_id
            )
        names = ("cont_out", "canid_logits", "nbr_pred", "z", "kl_per_node", "edge_logits")
        for name, t in zip(names, out):
            if isinstance(t, torch.Tensor):
                # `isnan(t).sum()` allocates a [N]→int64 reduction (~4GB for
                # 540M elements) → OOMs during the dump itself. Use scalar
                # reductions only: any/all return 0-dim, no big intermediate.
                diag[f"{name}_has_nan"] = bool(torch.isnan(t).any().item())
                diag[f"{name}_has_inf"] = bool(torch.isinf(t).any().item())
                diag[f"{name}_finite"] = not (diag[f"{name}_has_nan"] or diag[f"{name}_has_inf"])
                diag[f"{name}_absmax"] = (
                    float(t.abs().nan_to_num(neginf=0).max().item()) if t.numel() else 0.0
                )
                diag[f"{name}_shape"] = list(t.shape)
    log.error("nan_debug_intermediates", **diag)


def _measure_fwd_bwd(model, step_fn, batch, dev, *, debug_tag: str | None = None) -> int:
    """Run one forward/backward pass and return peak allocator bytes."""
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    # Snapshot RNG immediately before the forward so a NaN can be replayed.
    pre_cpu_state = torch.get_rng_state()
    pre_cuda_state = torch.cuda.get_rng_state(dev)
    try:
        _loss(step_fn(batch)).backward()
    except ValueError as e:
        if "non-finite" in str(e) and debug_tag is not None:
            log.error(
                "nan_replay_seed",
                tag=debug_tag,
                cpu_state_sha=hash(bytes(pre_cpu_state.numpy().tobytes()[:32])),
                cuda_state_sha=hash(bytes(pre_cuda_state.cpu().numpy().tobytes()[:32])),
                note="restore via torch.set_rng_state + torch.cuda.set_rng_state",
            )
            _dump_intermediates(
                model,
                batch,
                tag=debug_tag,
                cpu_state=pre_cpu_state,
                cuda_state=pre_cuda_state,
                dev=dev,
            )
        raise
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(dev)
    return int(torch.cuda.max_memory_allocated(dev))


def _measure_fwd(model, step_fn, batch, dev) -> tuple[int, float]:
    """Run one forward pass under ``no_grad`` and return peak bytes + wall time."""
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        _loss(step_fn(batch))
    torch.cuda.synchronize(dev)
    return int(torch.cuda.max_memory_allocated(dev)), time.perf_counter() - t0


def probe(
    model,
    train_dataset,
    *,
    quadratic: bool = False,
    min_steps: int | None = None,
) -> BudgetResult:
    """Estimate a packing budget from actual training graphs."""
    if not torch.cuda.is_available() or model is None or train_dataset is None:
        raise RuntimeError("budget probe requires CUDA + model + train_dataset")

    # Local import: pack_offline lives in datamodule, importing at module
    # level would create a cycle if anything in datamodule imports budget.
    from graphids.core.data.datamodule.sampler import pack_offline

    dev = model.device
    # Fork CPU + CUDA RNG so probe's torch.rand (masker) and torch.randn_like
    # (VGAE reparam eps) don't consume entropy training will need afterward,
    # and so the probe is bit-deterministic across re-runs (same seed → same
    # draw → same NaN if any). Required for replay-based NaN debugging.
    rng_devices = [dev.index] if dev.index is not None else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(_PROBE_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_PROBE_SEED)
        return _probe_body(
            model, train_dataset, dev, pack_offline, quadratic=quadratic, min_steps=min_steps
        )


def _target_bytes() -> int:
    if torch.cuda.is_available():
        try:
            free = int(torch.cuda.mem_get_info()[0])
        except Exception:  # pragma: no cover - defensive around mocked CUDA
            free = _DEFAULT_TARGET_BYTES
    else:
        free = _DEFAULT_TARGET_BYTES
    return max(1, int(free * _SAFETY))


def _dataset_size_stats(train_dataset) -> tuple[int, int, int, float]:
    if train_dataset is None:
        return 1, 1, 0, _DEFAULT_EDGES_PER_NODE
    sizes: list[int] = []
    edge_sizes: list[int] = []
    for graph in train_dataset:
        sizes.append(int(graph.num_nodes))
        edge_sizes.append(int(graph.num_edges))
    if not sizes:
        raise RuntimeError("budget heuristic: train_dataset is empty")
    total_nodes = sum(sizes)
    total_edges = sum(edge_sizes)
    epn = total_edges / max(1, total_nodes)
    return max(sizes), max(edge_sizes), total_nodes, max(epn, 1.0)


def _heuristic_budget(
    dataset: str,
    *,
    train_dataset=None,
    quadratic: bool = False,
    heads: int | None = None,
    min_steps: int | None = None,
    binding: str = "heuristic",
) -> BudgetResult:
    """Return a deterministic node/edge budget without running the model.

    This intentionally solves the robust packing problem, not exact model
    memory prediction. The empirical probe is still preferred when CUDA, the
    model, and the train dataset are all available.
    """
    target = _target_bytes()
    max_nodes, max_edges, total_nodes, epn = _dataset_size_stats(train_dataset)
    reserve = int(target * _CUDNN_RESERVE)
    usable = max(1, target - reserve)

    if quadratic:
        head_factor = max(1.0, float(heads or 1) / 4.0)
        budget = int(math.sqrt(usable / (_HEURISTIC_GPS_BPN2 * head_factor)))
    else:
        per_node = _HEURISTIC_BPN + int(epn * _HEURISTIC_BPE)
        budget = int(usable / max(1, per_node))
    budget = max(max_nodes, budget, 1)

    if min_steps is not None and min_steps > 1 and total_nodes > 0:
        step_cap = total_nodes // min_steps
        if step_cap > max_nodes:
            budget = min(budget, step_cap)

    edge_budget = max(max_edges, int(budget * epn * _EPN_HEADROOM), 1)
    log.info(
        "budget_heuristic",
        dataset=dataset,
        quadratic=quadratic,
        target_mb=target // _MB,
        budget_nodes=budget,
        budget_edges=edge_budget,
        max_nodes=max_nodes,
        max_edges=max_edges,
        edges_per_node=round(epn, 2),
        binding=binding,
    )
    return BudgetResult(
        budget=budget,
        edge_budget=edge_budget,
        binding=binding,
        target_bytes=target,
    )


def _can_probe(model, train_dataset) -> bool:
    return torch.cuda.is_available() and model is not None and train_dataset is not None


def _probe_body(
    model, train_dataset, dev, pack_offline, *, quadratic: bool, min_steps: int | None
) -> BudgetResult:
    # forward() returns outputs, not a scalar loss; route through training_step.
    step_fn = getattr(model, "_step", None) or (lambda b: model.training_step(b, 0))
    was_training = model.training

    # ── 1. Per-graph sizes (offline, no GPU work) ──────────────────────
    sizes_list = [int(g.num_nodes) for g in train_dataset]
    edge_sizes_list = [int(g.num_edges) for g in train_dataset]
    if not sizes_list:
        raise RuntimeError("budget probe: train_dataset is empty")
    sizes_t = torch.tensor(sizes_list, dtype=torch.long)
    edges_t = torch.tensor(edge_sizes_list, dtype=torch.long)

    # ── 2. B0: max single-graph size. pack_offline drops anything over its
    # max_num/max_edges, so B0 = max guarantees zero drops, and the largest
    # packed batch is at least the largest single graph (real probe scale).
    B0_nodes = int(sizes_t.max().item())
    B0_edges = int(edges_t.max().item())
    plans_0 = pack_offline(sizes_t, max_num=B0_nodes, edge_sizes=edges_t, max_edges=B0_edges)
    if not plans_0:
        raise RuntimeError("pack_offline returned 0 plans under B0 budget")

    # Build candidate set: argmax-V batch and argmax-E batch from plans_0.
    plan_V = torch.tensor([int(sizes_t[plan].sum().item()) for plan in plans_0])
    plan_E = torch.tensor([int(edges_t[plan].sum().item()) for plan in plans_0])
    candidate_indices = sorted({int(plan_V.argmax()), int(plan_E.argmax())})

    # ── 3. Warmup + baseline measurement, then probe each candidate ────
    model.train()
    fwd_peak = 0
    fwd_time = 0.0
    candidate_peaks: list[tuple[int, int, int]] = []  # (V, E, peak)

    with torch.enable_grad(), _silent_log(model):
        # Warmup on the first candidate so cuDNN has primed at least one shape.
        warm_batch = Batch.from_data_list(
            [train_dataset[i] for i in plans_0[candidate_indices[0]]]
        ).to(dev)
        for _ in range(3):
            _loss(step_fn(warm_batch)).backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        baseline = torch.cuda.memory_allocated(dev)
        # Measure fwd-only timing on the warm batch (used by autosize_workers).
        model.eval()
        fwd_peak, fwd_time = _measure_fwd(model, step_fn, warm_batch, dev)
        model.train()
        del warm_batch
        torch.cuda.empty_cache()

        # Probe each candidate (fwd+bwd peak).
        for ci in candidate_indices:
            cb = Batch.from_data_list([train_dataset[i] for i in plans_0[ci]]).to(dev)
            cV, cE = int(cb.num_nodes), int(cb.num_edges)
            cpeak = _measure_fwd_bwd(model, step_fn, cb, dev, debug_tag=f"candidate_ci{ci}")
            candidate_peaks.append((cV, cE, cpeak))
            del cb
            torch.cuda.empty_cache()

    worst_V, worst_E, worst_peak = max(candidate_peaks, key=lambda p: p[2])

    # ── 4. Resident-state subtract ────────────────────────────────────
    free = torch.cuda.mem_get_info()[0]
    target = max(1, int(free * _SAFETY))
    # Probe runs BEFORE Lightning's configure_optimizers — Adam state (m,v
    # fp32) materializes on first optimizer.step. Subtract 2×params bytes.
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    optim_overhead = 2 * param_bytes
    cudnn_reserve = int(free * _CUDNN_RESERVE)
    headroom = max(1, target - baseline - optim_overhead - cudnn_reserve)

    activation = max(1, worst_peak - baseline)
    bwd_mult = max(1.0, worst_peak / max(1, fwd_peak))

    # ── 5. Derive B1 from MEASURED per-node at REAL scale ─────────────
    if quadratic:
        alpha = activation / (worst_V * worst_V)
        B1 = max(B0_nodes, int(math.sqrt(headroom / alpha)))
    else:
        per_node = activation / worst_V
        B1 = max(B0_nodes, int(headroom / per_node))

    # Cap B1 to enforce a minimum step count per epoch. B0_nodes is the floor
    # (can't pack smaller than the largest single graph), so the cap only bites
    # when the VRAM budget would produce fewer steps than requested.
    if min_steps is not None and min_steps > 1:
        total_nodes = sum(sizes_list)
        step_cap = total_nodes // min_steps
        if step_cap > B0_nodes:
            B1 = min(B1, step_cap)

    epn = worst_E / max(1, worst_V)
    edge_budget = max(B0_edges, int(B1 * epn * _EPN_HEADROOM))

    # ── 6. Repack with B1 (if it grew) and sanity-probe new largest ───
    repack_done = False
    sanity_V = sanity_E = sanity_peak = 0
    if B1 > B0_nodes:
        repack_done = True
        plans_1 = pack_offline(sizes_t, max_num=B1, edge_sizes=edges_t, max_edges=edge_budget)
        if not plans_1:
            raise RuntimeError("pack_offline returned 0 plans under B1 budget")
        plan1_V = torch.tensor([int(sizes_t[plan].sum().item()) for plan in plans_1])
        sci = int(plan1_V.argmax())
        sb = Batch.from_data_list([train_dataset[i] for i in plans_1[sci]]).to(dev)
        sanity_V, sanity_E = int(sb.num_nodes), int(sb.num_edges)
        with torch.enable_grad(), _silent_log(model):
            sanity_peak = _measure_fwd_bwd(model, step_fn, sb, dev, debug_tag="sanity")
        del sb
        torch.cuda.empty_cache()
        if sanity_peak > target:
            model.train(was_training)
            raise RuntimeError(
                f"budget probe: post-repack sanity probe exceeded target. "
                f"V={sanity_V} E={sanity_E} peak={sanity_peak // _MB}MB "
                f"target={target // _MB}MB free={free // _MB}MB. "
                f"Lower max budget (e.g. set GRAPHIDS_BUDGET_SAFETY_MARGIN<{_SAFETY}), "
                f"reduce window/dataset, or use a larger GPU."
            )

    model.train(was_training)

    log.info(
        "budget_probed",
        quadratic=quadratic,
        free_mb=free // _MB,
        baseline_mb=baseline // _MB,
        worst_V=worst_V,
        worst_E=worst_E,
        worst_peak_mb=worst_peak // _MB,
        activation_mb=activation // _MB,
        optim_overhead_mb=optim_overhead // _MB,
        cudnn_reserve_mb=cudnn_reserve // _MB,
        target_mb=target // _MB,
        budget_nodes=B1,
        budget_edges=edge_budget,
        edges_per_node=round(epn, 2),
        bwd_mult=round(bwd_mult, 2),
        t_fwd_ms=round(fwd_time * 1000, 1),
        repacked=repack_done,
        sanity_V=sanity_V,
        sanity_peak_mb=sanity_peak // _MB,
        n_candidates=len(candidate_peaks),
    )
    return BudgetResult(
        budget=B1,
        edge_budget=edge_budget,
        backward_multiplier=bwd_mult,
        t_fwd=float(fwd_time),
        target_bytes=target,
    )


def node_budget(
    dataset: str,
    *,
    model=None,
    train_dataset=None,
    conv_type: str | None = None,
    heads: int | None = None,
    min_steps: int | None = None,
) -> BudgetResult:
    if conv_type is None and model is not None:
        conv_type = getattr(model.hparams, "conv_type", "gatv2")
    quadratic = conv_type == "gps"
    mode = os.environ.get("GRAPHIDS_BUDGET_MODE", "auto").lower()
    strict_probe = os.environ.get("GRAPHIDS_BUDGET_STRICT_PROBE", "0") == "1"
    if mode in {"probe", "measured", "auto"} and _can_probe(model, train_dataset):
        try:
            return probe(model, train_dataset, quadratic=quadratic, min_steps=min_steps)
        except Exception:
            if strict_probe or mode in {"probe", "measured"}:
                raise
            log.warning("budget_probe_failed_using_heuristic", dataset=dataset, exc_info=True)
            return _heuristic_budget(
                dataset,
                train_dataset=train_dataset,
                quadratic=quadratic,
                heads=heads,
                min_steps=min_steps,
                binding="probe_failed_heuristic",
            )
    if mode in {"probe", "measured"} and strict_probe:
        raise RuntimeError("budget probe requires CUDA + model + train_dataset")
    return _heuristic_budget(
        dataset,
        train_dataset=train_dataset,
        quadratic=quadratic,
        heads=heads,
        min_steps=min_steps,
    )


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
