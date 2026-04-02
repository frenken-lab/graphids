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

## Dagster Retry Note

`RetryPolicy(max_retries=2, delay=30)` — 3 total attempts per asset per run.
Once exhausted, recovery requires a new `dg launch` or manual `sbatch` for leaf nodes.
