# VGAE cuBLAS fp32 overflow on V100 — strict-precision workaround

**Date:** 2026-05-06 · **Hardware:** OSC Pitzer V100 16GB · **Build:** PyTorch 2.8 + CUDA 12.6
**Fix commit:** `e323038` (strict-precision wired in `orchestrate.py::_ensure_runtime`)

## TL;DR

`nn.Linear` at matmul `[300793, 64] @ [64, 1791]` returned fp32-saturated / NaN
on V100 from finite inputs (`z` absmax=11.68, Kaiming `|W| ≤ 0.125` → algebraic
bound ~93; observed `3.40282347e+38` = `finfo(float32).max`). **cuBLAS defect on
Volta** — heuristics pick kernels with reduced-precision intermediate accumulation
for fp32 GEMM at this shape. Not a graphids bug. Ampere/Hopper unaffected.

## Workaround (shipped, permanent)

`graphids/orchestrate.py::_ensure_runtime`:

```python
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

Forces fp32 accumulation. Verified bit-deterministic: same
`GRAPHIDS_PROBE_SEED=20260506` → NaN without flags, clean with. Full fit
completed (jid 47317870, 11:56). Negligible cost on Volta; no-op on Ampere/Hopper.

## Probe instrumentation (pattern for future NaN debugging) — `graphids/core/budget.py`

- `torch.random.fork_rng()` — probe RNG doesn't pollute training draws.
- `GRAPHIDS_PROBE_SEED` env var — reproducible; vary to bisect.
- Snapshot CPU+CUDA RNG before each fwd+bwd; on non-finite loss, restore and
  replay through `_dump_intermediates` (exact replay — re-rolls give
  contradictory `_finite=True` + `_absmax=NaN`).
- Use `isnan(t).any()`/`isinf(t).any()`. `.sum()` allocates `[N]→int64` ~4 GB
  on 540M-element `canid_logits` and OOMs the dump on V100.
- `bad_params: []` (`isfinite(p).all()` over named params) rules out weight
  corruption → points at the matmul kernel, not upstream numerics.

## Hardware gotchas

- **V100**: hits this defect at large-M GEMM; strict flags required.
  Separate vectorized-gather race on set_01/04 → `CUDA_LAUNCH_BLOCKING=1`.
- **A100 / H100**: unaffected; flags are no-ops.

## Reproduce: `gx run ablations.supervised -d set_03 -s 42 --filter 'vgae*' -o /tmp/v.json && gx plans submit --plan /tmp/v.json -C cardinal` (pre-fix: NaN ~3 min in).
