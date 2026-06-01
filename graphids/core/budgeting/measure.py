"""CUDA measurement primitives used by empirical budget probes."""

from __future__ import annotations

import time

import torch

from .diagnostics import _dump_intermediates, _loss


def _measure_fwd_bwd(model, step_fn, batch, dev, *, debug_tag: str | None = None) -> int:
    """Run one forward/backward pass and return peak allocator bytes."""
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    pre_cpu_state = torch.get_rng_state()
    pre_cuda_state = torch.cuda.get_rng_state(dev)
    try:
        _loss(step_fn(batch)).backward()
    except ValueError as e:
        if "non-finite" in str(e) and debug_tag is not None:
            from structlog import get_logger

            get_logger(__name__).error(
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
