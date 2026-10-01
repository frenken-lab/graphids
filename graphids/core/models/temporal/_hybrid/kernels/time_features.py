"""Fixed sinusoidal time features; learned projections stay in PyTorch.

These wrappers intentionally accept only float32 CUDA inputs without gradients.
The time encoder's frequencies are buffers, so no backward is needed for the
normal timestamp/IAT path. Features retain PyTorch's sine-then-cosine order.
"""

from __future__ import annotations

import torch

from .rhythm import tl, triton

if triton is not None:

    @triton.jit
    def _encode(delta, frequency):
        # log(1+x) loses small positive IATs; this correction preserves log1p
        # accuracy when rounding 1+x would otherwise discard those bits.
        x = tl.maximum(delta, 0)
        rounded = 1 + x
        logged = tl.log(rounded) + (x - (rounded - 1)) / rounded
        phase = logged / frequency
        return tl.sin(phase), tl.cos(phase)

    @triton.jit
    def _time_kernel(Values, Freqs, Out, N: tl.constexpr, WIDTH: tl.constexpr,
                     DIM: tl.constexpr, HALF: tl.constexpr, ROW: tl.constexpr,
                     COL: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = index < N * WIDTH * DIM
        channel = index % DIM
        value_pos = index // DIM
        value = tl.load(Values + value_pos // WIDTH * ROW + value_pos % WIDTH * COL, mask, 0)
        freq = tl.load(Freqs + channel % HALF, mask, 1)
        sine, cosine = _encode(value, freq)
        tl.store(Out + index, tl.where(channel < HALF, sine, cosine), mask)

    @triton.jit
    def _memory_time_kernel(Last, Src, Dst, Now, Freqs, Out, N: tl.constexpr,
                            DIM: tl.constexpr, HALF: tl.constexpr, SS: tl.constexpr,
                            DS: tl.constexpr, TS: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = index < N * 2 * DIM
        channel = index % DIM
        endpoint = index // DIM % 2
        lane = index // (2 * DIM)
        src = tl.load(Src + lane * SS, mask, 0)
        dst = tl.load(Dst + lane * DS, mask, 0)
        last = tl.load(Last + tl.where(endpoint == 0, src, dst), mask, -1)
        now = tl.load(Now + lane * TS, mask, 0)
        delta = tl.where(last >= 0, now - last, 0)
        freq = tl.load(Freqs + channel % HALF, mask, 1)
        sine, cosine = _encode(delta, freq)
        tl.store(Out + index, tl.where(channel < HALF, sine, cosine), mask)


def _validate(values, freqs, dim):
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if (not values.is_cuda or values.dtype != torch.float32 or values.requires_grad
            or freqs.device != values.device or freqs.dtype != torch.float32
            or freqs.requires_grad or not freqs.is_contiguous()
            or freqs.shape != (max(1, (dim + 1) // 2),) or dim < 1):
        raise ValueError("expected CUDA float32 values and fixed contiguous frequency buffers")


def time_features_triton(values, freqs, dim: int) -> torch.Tensor:
    """Encode a possibly strided [active_lanes, history] IAT snapshot.

    The returned [active_lanes, history * dim] tensor owns its storage, so
    subsequent history mutation cannot alter projection inputs or backward.
    """
    _validate(values, freqs, dim)
    if values.ndim != 2:
        raise ValueError("time values must be [lanes, history]")
    n, width = values.shape
    out = values.new_empty((n, width * dim))
    if out.numel():
        with torch.cuda.device(values.device):
            _time_kernel[(triton.cdiv(out.numel(), 128),)](
                values, freqs, out, n, width, dim, freqs.numel(), *values.stride(),
                BLOCK=128, enable_fp_fusion=False,
            )
    return out


def memory_time_features_triton(last_seen, flat_src, flat_dst, now, freqs, dim: int) -> torch.Tensor:
    """Gather timestamps and encode elapsed time for both event endpoints.

    Unseen IDs (last_seen < 0) encode zero elapsed time. Negative elapsed time
    also clamps to zero. This function does not update timestamps or gather
    trainable memory rows; their causal update and autograd stay in PyTorch.
    """
    _validate(last_seen, freqs, dim)
    n = flat_src.numel()
    if last_seen.ndim != 1 or not last_seen.is_contiguous():
        raise ValueError("last_seen must be a contiguous flat vector")
    for vector, dtype in ((flat_src, torch.int64), (flat_dst, torch.int64), (now, torch.float32)):
        if (vector.device != last_seen.device or vector.shape != (n,)
                or vector.dtype != dtype or vector.requires_grad):
            raise ValueError("timestamp lane vectors must match CUDA state and have no gradients")
    out = last_seen.new_empty((n, 2 * dim))
    if out.numel():
        with torch.cuda.device(last_seen.device):
            _memory_time_kernel[(triton.cdiv(out.numel(), 128),)](
                last_seen, flat_src, flat_dst, now, freqs, out, n, dim, freqs.numel(),
                flat_src.stride(0), flat_dst.stride(0), now.stride(0), BLOCK=128, enable_fp_fusion=False,
            )
    return out
