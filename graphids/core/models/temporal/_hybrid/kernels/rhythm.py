"""Float32 rhythm gather, context features and Welford update for disjoint lanes."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rhythm_kernel(State, Src, Dst, Iat, Same, Features, N: tl.constexpr,
                       SRC_STRIDE: tl.constexpr, DST_STRIDE: tl.constexpr,
                       IAT_STRIDE: tl.constexpr, SAME_STRIDE: tl.constexpr,
                       MotifIds, MotifIats, Lanes, DestIds,
                       LENGTH: tl.constexpr, LANE_STRIDE: tl.constexpr, DEST_STRIDE: tl.constexpr,
                       BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = lane < N
        src = tl.load(Src + lane * SRC_STRIDE, mask, 0) * 3
        dst = tl.load(Dst + lane * DST_STRIDE, mask, 0) * 3
        same = tl.load(Same + lane * SAME_STRIDE, mask, False)
        value = tl.maximum(tl.load(Iat + lane * IAT_STRIDE, mask, 0), 0)
        sc = tl.load(State + src, mask, 0)
        sm = tl.load(State + src + 1, mask, 0)
        ss = tl.load(State + src + 2, mask, 0)
        dc = tl.load(State + dst, mask, 0)
        dm = tl.load(State + dst + 1, mask, 0)
        ds = tl.load(State + dst + 2, mask, 0)
        scc = tl.maximum(sc, 0)
        dcc = tl.maximum(dc, 0)
        sv = tl.where(scc > 1, ss / tl.maximum(scc - 1, 1), 0)
        dv = tl.where(dcc > 1, ds / tl.maximum(dcc - 1, 1), 0)
        tl.store(Features + lane * 6, tl.log(1 + scc), mask)
        tl.store(Features + lane * 6 + 1, sm, mask)
        tl.store(Features + lane * 6 + 2, tl.sqrt(tl.maximum(sv, 0)), mask)
        tl.store(Features + lane * 6 + 3, tl.log(1 + dcc), mask)
        tl.store(Features + lane * 6 + 4, dm, mask)
        tl.store(Features + lane * 6 + 5, tl.sqrt(tl.maximum(dv, 0)), mask)
        sn = sc + 1
        delta = value - sm
        mn = sm + delta / sn
        vn = ss + delta * (value - mn)
        dn = tl.where(same, sn, dc) + 1
        base_mean = tl.where(same, mn, dm)
        delta = value - base_mean
        dmn = base_mean + delta / dn
        dvn = tl.where(same, vn, ds) + delta * (value - dmn)
        # A self-edge stores only the final (second) update, avoiding aliasing stores.
        tl.store(State + src, sn, mask & ~same)
        tl.store(State + src + 1, mn, mask & ~same)
        tl.store(State + src + 2, vn, mask & ~same)
        tl.store(State + dst, dn, mask)
        tl.store(State + dst + 1, dmn, mask)
        tl.store(State + dst + 2, dvn, mask)

        if LENGTH > 0:
            # Each vector element owns a lane. Shift left in order, then append;
            # source history at position+1 is read before that position is written.
            owner = tl.load(Lanes + lane * LANE_STRIDE, mask, 0)
            row = owner * LENGTH
            for position in tl.static_range(LENGTH - 1):
                old_id = tl.load(MotifIds + row + position + 1, mask, 0)
                old_iat = tl.load(MotifIats + row + position + 1, mask, 0)
                tl.store(MotifIds + row + position, old_id, mask)
                tl.store(MotifIats + row + position, old_iat, mask)
            dest_id = tl.load(DestIds + lane * DEST_STRIDE, mask, 0)
            raw_iat = tl.load(Iat + lane * IAT_STRIDE, mask, 0)
            tl.store(MotifIds + row + LENGTH - 1, dest_id, mask)
            # Match the lane writer, which stores its supplied IAT unchanged.
            tl.store(MotifIats + row + LENGTH - 1, raw_iat, mask)


def rhythm_features_update_triton(
    rhythm_flat, flat_src, flat_dst, iat, same_id, *,
    motif_ids=None, motif_iats=None, lane_idx=None, dst_idx=None,
) -> torch.Tensor:
    """Return pre-update features and mutate float32 state on CUDA.

    Each active lane must own disjoint rows; only source/destination aliasing
    within one lane is supported. Indices must be in bounds and ``same_id``
    must agree with their equality. Repeated IDs across successive calls are safe.
    Strided lane vectors (such as columns of a chunk) are supported.
    Optional motif state is shifted/appended in the same launch. Callers must
    snapshot motif history first: context and embedding backward need old IDs.
    """
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    vectors = (flat_src, flat_dst, iat, same_id)
    if (not rhythm_flat.is_cuda or rhythm_flat.dtype != torch.float32
            or rhythm_flat.ndim != 2 or rhythm_flat.shape[1] != 3
            or not rhythm_flat.is_contiguous() or rhythm_flat.requires_grad):
        raise ValueError("rhythm state must be contiguous CUDA float32 [rows, 3] without gradients")
    n = flat_src.numel()
    if any(v.device != rhythm_flat.device or v.ndim != 1 or v.numel() != n for v in vectors):
        raise ValueError("lane vectors must have matching shapes and share the state CUDA device")
    if (flat_src.dtype != torch.int64 or flat_dst.dtype != torch.int64
            or iat.dtype != torch.float32 or same_id.dtype != torch.bool or iat.requires_grad):
        raise ValueError("expected int64 indices, float32 IAT without gradients, and bool same_id")
    length = 0
    optional = (motif_ids, motif_iats, lane_idx, dst_idx)
    if any(v is not None for v in optional):
        if any(v is None for v in optional):
            raise ValueError("motif fusion requires both history tensors and lane/destination indices")
        if (motif_ids.ndim != 2 or motif_iats.shape != motif_ids.shape or motif_ids.shape[1] < 1
                or not motif_ids.is_contiguous() or not motif_iats.is_contiguous()
                or motif_ids.dtype != torch.int64 or motif_iats.dtype != torch.float32
                or motif_iats.requires_grad
                or any(v.device != rhythm_flat.device for v in optional)
                or any(v.shape != (n,) or v.dtype != torch.int64 for v in (lane_idx, dst_idx))):
            raise ValueError("motif fusion requires contiguous CUDA history and int64 lane vectors")
        length = motif_ids.shape[1]
    else:
        # Unused pointers are eliminated by the LENGTH=0 specialization.
        motif_ids = motif_iats = lane_idx = dst_idx = flat_src
    features = rhythm_flat.new_empty((n, 6))
    if n:
        with torch.cuda.device(rhythm_flat.device):
            _rhythm_kernel[(triton.cdiv(n, 128),)](
                rhythm_flat, flat_src, flat_dst, iat, same_id, features, n,
                *(v.stride(0) for v in vectors),
                motif_ids, motif_iats, lane_idx, dst_idx, length, lane_idx.stride(0), dst_idx.stride(0), BLOCK=128, enable_fp_fusion=False,
            )
    return features
