# Ablation Run Bug Log

> Created: 2026-04-02 (session 8). All 8 bugs fixed.

## Pattern Analysis (apply to future runs)

**Pattern A — Stale tests after code changes (bugs 1-3):**
Tests hardcoded old values from before sessions 6-7. Code was correct, tests drifted.
Prevention: run `scripts/submit.sh tests` after any code change before launching runs.

**Pattern B — Stale artifacts from prior code structure (bugs 5-6):**
Config YAML copied from another model without removing model-specific fields.
Old checkpoints with class_paths from pre-reorganization module layout.
Prevention: `orchestrate validate` catches YAML parse errors. Clear stale checkpoints
when module paths change.

**Pattern C — Identity keys ≠ model keys (bugs 7-8):**
Identity keys (for hash dedup) were blindly passed as `--model.init_args.X` CLI overrides.
Not all identity keys are valid model init params. Prevention: `model_keys` (subset of
`identity_keys`) controls which keys become CLI overrides.

## Bug Summary

| # | Bug | Root cause | Pattern |
|---|-----|-----------|---------|
| 1-3 | Stale test assertions | Tests not updated after sessions 6-7 | A |
| 4 | FusionDataModule missing `test_dataloader` | Standalone | — |
| 5 | DGI `base.yaml` has spurious `auxiliaries` field | Copied from VGAE | B |
| 6 | Old checkpoints with stale class_path | Pre-reorg module paths | B |
| 7 | DGI gets `variational` override it can't accept | identity_keys ≠ model_keys | C |
| 8 | Conv-variant autoencoders use wrong model config | `model_type` propagation | C |

## Bug 9 — DGI torch.compile inductor crash (autoencoder_c479d625)

**Status:** Open — blocks entire DGI branch (autoencoder + downstream normal/curriculum/fusion)

**Symptom:** DGI autoencoder fails during sanity check with `BackendCompilerFailed`:
```
RuntimeError: Not all inputs to pattern found in match.kwargs.
Perhaps one of the inputs is unused? argnames=['x', 'slice_shape'], match.kwargs={'x': arg2_1}
```

**Root cause:** `torch.compile(self.model, dynamic=True)` at `dgi.py:191` compiles the
`GraphInfomaxModel`. The inductor backend's pattern matcher fails on the DGI graph
structure during the VRAM budget probe (`_probe_bytes_per_node` → `model(batch)`).
The crash happens inside `joint_graph_passes` → `pattern_matcher.py:1961`.

This is a PyTorch inductor bug (torch 2.8.0) specific to the DGI model's graph pattern.
GAT/VGAE models compile fine — DGI's `InfomaxModel` structure (dual encoder + summary +
discriminator) likely creates an unusual FX graph that triggers the pattern match failure.

**Call chain:** `val_dataloader()` → `_build_loader()` → `vram_node_budget()` →
`_probe_bytes_per_node()` → `model(batch)` → compiled forward → inductor crash

**Fix:** Set `compile_model: false` in `graphids/config/models/dgi/base.yaml`. DGI's
24.6K params don't benefit from torch.compile anyway. Or guard with
`torch._dynamo.reset()` before probe.

**Pattern:** New — torch.compile incompatibility. Add to smoke test: verify all model
types survive one forward pass under compile.

**Log evidence:** `slurm_logs/dgi_manual_46265877.err` lines 35-169.

## Bug 10 — normal_ab6a75a4 phantom resume_ckpt (fixed in code, stale orchestrator)

**Status:** Fixed in code (commit ebd7e1f), but running orchestrator uses old code

**Symptom:** `normal_ab6a75a4` fails in 13 seconds (exit 2) on every attempt. Orchestrator
log shows `ckpt_path` override pointing to a non-existent `last.ckpt`.

**Root cause:** The old orchestrator code (pre-ebd7e1f) injected `resume_ckpt` overrides
from the orchestrator node without verifying the checkpoint exists on the worker. For
ab6a75a4, the checkpoint directory doesn't exist:
```
/fs/ess/PAS1266/kd-gat/dev/rf15/set_01/gat_small_normal_ab6a75a4/seed_42/checkpoints/
→ No such file or directory
```
Lightning receives `--ckpt_path=<nonexistent>` and exits immediately with code 2.

**Fix (already applied):** Commit ebd7e1f moved auto-resume to `train_entrypoint.py:39-45`
where it checks `last_ckpt.exists()` on the actual worker node. The running orchestrator
(job 46260678, started before ebd7e1f) still uses old code.

**Recovery:** Manually resubmit ab6a75a4 (dagster retries exhausted). Or restart the
orchestrator to pick up the fix. The downstream curriculum + fusion assets are blocked.

**Pattern:** B — stale code running in long-lived SLURM job doesn't pick up fixes.
Prevention: restart orchestrator after code changes, or use exec-based reload.

## Dagster Retry Note

`RetryPolicy(max_retries=2, delay=30)` — 3 total attempts per asset per run.
Once exhausted, recovery requires a new `dg launch` or manual `sbatch` for leaf nodes.
