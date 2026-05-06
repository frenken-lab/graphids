# VGAE budget probe — fp32 overflow in `nn.Linear` on V100

**Date:** 2026-05-06 (UTC ~01:50)
**Hardware:** OSC Pitzer V100 16GB
**Build:** PyTorch 2.8 + CUDA 12.6, model state `git_sha=05f8a59860bb`
**Plan:** `ablations.supervised --filter 'vgae*' -d set_03 -s 42` (VGAE on can-train-and-test-v1.5 set_03, `label_filter='benign'`)

## TL;DR

`nn.Linear` produces fp32-overflow (`absmax=3.40282347e+38` = `torch.finfo(torch.float32).max`) and NaN/Inf outputs when applied to a finite latent `z` (absmax=11.68) at matmul shape `[300793, 64] @ [64, 1791]` on V100. There is no algebraic path from inputs of that magnitude to fp32 max — this is a **cuBLAS-level numerical defect at this specific (M, N, K) on Volta**, not a model bug.

## Reproducible diagnostic

After patching the budget probe to:
- isolate RNG via `torch.random.fork_rng()` + fixed seed `GRAPHIDS_PROBE_SEED=20260506`
- snapshot CPU + CUDA RNG state immediately before the failing fwd+bwd
- replay the failing forward via `torch.set_rng_state` / `torch.cuda.set_rng_state`

we get bit-deterministic NaN every run. Sample `nan_debug_intermediates` log line:

```json
{
  "tag": "sanity",
  "V": 300793,
  "E": 652974,
  "bad_params": [],
  "z_finite": true,           "z_absmax": 11.68,
  "cont_out_finite": true,    "cont_out_absmax": 6.97,
  "kl_per_node_finite": true, "kl_per_node_absmax": 1.11,
  "canid_logits_has_nan": true, "canid_logits_has_inf": true,
    "canid_logits_absmax": NaN,
    "canid_logits_shape": [300793, 1791],
  "nbr_logits_has_nan": true,   "nbr_logits_has_inf": true,
    "nbr_logits_absmax": 3.40282347e+38,
    "nbr_logits_shape": [300793, 1791]
}
```

`bad_params: []` rules out weight corruption (`isfinite(p).all()` over every named parameter). `z` is healthy. The single `Linear(64→1791)` (`canid_classifier`) and the 3-layer MLP (`neighborhood_decoder`) on the same z both produce non-finite outputs — pointing at the matmul kernel itself rather than any model-side numerics.

## Why this is *not* algebraic

- `nn.Linear` default init: Kaiming uniform with bound `sqrt(1/fan_in)` for fan_in=64 ⇒ `|W| ≤ 0.125`.
- Per-output element of `Linear(64, 1791)`: `out[i,j] = Σ_k W[j,k]·z[i,k] + b[j]`.
- Worst case |out| ≤ 64 × 0.125 × 11.68 ≈ **93**. Reaching fp32 max (3.40e38) requires `>10^36×` amplification across a matmul that algebraically caps at ~93.
- The 3-layer MLP nominally amplifies further but is bounded by ReLU/Dropout(p=0.1) and same Kaiming weights — saturation impossible from real-valued accumulation.

## Hypothesis: cuBLAS GEMM numerical defect at Volta

V100 cuBLAS chooses an algorithm based on (M, N, K, dtype, layout). Some heuristics select kernels that use reduced-precision intermediate reductions even for fp32 inputs (e.g., split-K reductions accumulating to fp16 buffers, or pre-Ampere code paths with looser numerical guarantees). For `[300793, 64] @ [64, 1791]` there is at least one path that returns saturated/NaN for inputs in our range.

This is observable because:
- Smaller batches (the candidate probes at V=57) succeed with the same model and weights.
- The output saturates at exactly fp32 max — characteristic of accumulator overflow in a low-precision intermediate.
- `bad_params: []` and `z` finite at the entrance to the matmul rule out everything upstream.

## Workarounds

In order of preference:

1. **Cap the probe / pack budget so this matmul shape never appears.** With `max_num ≈ 64K`, the GEMM becomes `[64K, 64] @ [64, 1791]` — well inside cuBLAS-safe territory. Other ablations don't hit this because they have full label distribution and pack denser smaller batches naturally; VGAE's `label_filter='benign'` produces unusually large packed batches (105K benign graphs available; entire benign pool packs into a few large bins).
2. **Force strict-precision reductions:** set
   ```python
   torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
   torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
   ```
   before `pl.Trainer.fit`. Forces cuBLAS to keep accumulators in fp32. Worth one job to confirm; cleaner than capping budget.
3. **Move VGAE training to Cardinal H100 / Ascend A100.** Ampere/Hopper cuBLAS kernels are different code paths and typically don't exhibit this. (Requires logging into the right cluster — sbatch can only target the local cluster on OSC, see `reference/osc_gpu_clusters` memory.)

## Probe instrumentation in place (v5)

`graphids/core/budget.py` now:

- Wraps the probe in `torch.random.fork_rng()` so probe RNG consumption doesn't pollute the training-time draws — required for reproducibility AND for not silently shifting batch sampling order during fit.
- Seeds with `GRAPHIDS_PROBE_SEED` (default `20260506`) — same draw → same outcome across runs. Bisecting flaky NaN by varying this env var is the official debug workflow.
- Snapshots CPU + CUDA RNG state right before each fwd+bwd. On `ValueError` from `loss_fn` non-finite check, restores state and replays through `_dump_intermediates` for an exact-replay forward (no longer rolls fresh randomness — the prior `_dump_intermediates` was running a different draw than the one that failed, which is why earlier dumps showed contradictory `_finite=True` + `_absmax=NaN`).
- Uses `isnan(t).any()` / `isinf(t).any()` (scalar reductions) instead of `.sum()` (allocates `[N]→int64` ~4 GB for the 540M-element `canid_logits`, OOM'd the dump itself on V100).

## Reproducing the failure manually

```bash
source .env && source .venv/bin/activate
gx run ablations.supervised -d set_03 -s 42 --filter 'vgae*' -o /tmp/vgae.json
gx plans submit --plan /tmp/vgae.json -C cardinal   # routed to local cluster (pitzer)
# Fit fails ~3 min in at sanity probe with deterministic NaN. Stdout/stderr:
#   /fs/ess/PAS1266/graphids/dev/rf15/set_03/ablations/unsupervised/vgae/seed_42/.parsl_scripts/
```

To bisect non-failing seeds:
```bash
GRAPHIDS_PROBE_SEED=42 gx plans submit --plan /tmp/vgae.json -C cardinal
```

## Open

- Whether this affects VGAE training itself (post-probe, inside the fit loop) once the probe passes is **unconfirmed**. The previous "lucky" run trained to completion; the model checkpoint exists. May or may not have hit the same matmul issue silently and produced subtly bad gradients. Workaround #2 (now applied — see Update below) eliminates the matmul path entirely so this is moot going forward.
- No issue filed with PyTorch / cuBLAS yet. Should reproduce on a minimal `[300793, 64] @ [64, 1791]` matmul with random input/weight tensors before reporting.

## Update — 2026-05-06 ~02:01 UTC — workaround #2 confirmed

In `graphids/orchestrate.py::_ensure_runtime`:
```python
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

Resubmitted the same plan with the same `GRAPHIDS_PROBE_SEED=20260506` (so the
matmul shape, RNG draw, and inputs are bit-identical to the failing run). The
sanity probe now passes:

```
budget_probed: sanity_V=300793 sanity_peak_mb=6209 (no nan_debug, no nan_replay)
```

Bit-deterministic same draw → no NaN with strict reductions. The defect lives
in the cuBLAS code path that picks reduced-precision intermediate accumulation
for fp32 GEMM at this shape on Volta. Disabling it falls back to a strictly-fp32
accumulation kernel that handles the shape correctly.

**Permanent change.** Strict reductions stay on in `_ensure_runtime` for all
runs — they only marginally affect throughput on Volta, never affect Ampere or
Hopper meaningfully (those architectures' default kernels already accumulate
in fp32), and make Volta numerically uniform with the newer clusters. Net cost
is negligible; net benefit is one fewer source of intermittent NaN.

Workaround #1 (cap budget to ~64K) and #3 (move to Cardinal) not pursued.
