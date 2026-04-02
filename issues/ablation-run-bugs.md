# Ablation Run Bug Log

> Created: 2026-04-02 (session 8)
> Status: **Active** — set_01 ablation in progress

## Summary

8 bugs found across hcrl_sa smoke test and set_01 ablation launch. Two recurring
patterns identified. All fixed.

## Pattern Analysis

**Pattern A — Stale tests after code changes (bugs 1-3):**
Tests hardcoded old values/behavior from before sessions 6-7. Code was correct,
tests drifted. Fix: update assertions mechanically.

**Pattern B — Stale artifacts from prior code structure (bugs 5-6):**
Config YAML copied from another model without removing model-specific fields.
Old checkpoints with class_paths from pre-reorganization module layout.
Fix: remove stale fields, clear stale checkpoints.

**Pattern C — Identity keys ≠ model keys (bugs 7-8):**
Identity keys (used for hash dedup) were blindly passed as `--model.init_args.X`
CLI overrides. Not all identity keys are valid model init params (e.g. `variational`
is VGAE-only, `model_type` shouldn't be a CLI override). Fix: separate `model_keys`
from `identity_keys` in stage defs, add per-model filtering.

## Bugs

### Bug 1-3: Stale test assertions (hcrl_sa smoke test)

**Source:** `tests/config/test_merge_parity.py`, `tests/orchestrate/test_overrides.py`
**Commit:** `ffbeac3`

| # | Test | Error | Fix |
|---|------|-------|-----|
| 1 | `test_merge_parity` (3 params) | `Unrecognized arguments: fit` | Remove `"fit"` from args — `run=False` disables subcommands |
| 2 | `test_smoke_recipe_expands` | `max_epochs: '50' != '2'` | Update assertion to match current `smoke_test.yaml` |
| 3 | `test_missing_config_files_skipped` | `FileNotFoundError` raised | Assert `FileNotFoundError` — session 7 hardened `read_yaml` |

### Bug 4: FusionDataModule missing test_dataloader (hcrl_sa smoke test)

**Source:** `graphids/core/preprocessing/datamodule.py`
**Commit:** `ffbeac3`

All 4 fusion SLURM jobs completed training but failed the test phase:
`MisconfigurationException: test_dataloader must be implemented`.
`FusionDataModule` had `train_dataloader` and `val_dataloader` but no `test_dataloader`.
`CANBusDataModule` had all three. Fix: add `test_dataloader()` delegating to `val_dataloader()`.

Note: fusion missing `.analyze_complete` is by design (`ANALYSIS_SUPPORTED_MODELS = {vgae, gat}`).

### Bug 5: DGI base.yaml has spurious `auxiliaries` field (set_01 ablation)

**Source:** `graphids/config/models/dgi/base.yaml`
**Commit:** `e7ff54c`

DGI autoencoder failed at jsonargparse parse time: `Option 'auxiliaries' is not accepted`.
`dgi/base.yaml` was copied from `vgae/base.yaml` and still contained `auxiliaries: []`.
`DGIModule.__init__` doesn't accept `auxiliaries` (only `VGAEModule` does for KD).
Fix: remove the field from `dgi/base.yaml`.

### Bug 6: Old checkpoints with stale class_path (set_01 ablation)

**Source:** Pre-existing run directories on ESS
**Fix:** Cleared stale checkpoints + markers

3 normal jobs (2af9d630, 56cc5893, ab6a75a4) failed with:
`module 'graphids.core.models' has no attribute 'gat'`.
These dirs had checkpoints from before the module reorganization
(`graphids.core.models.gat.GATModule` → `graphids.core.models.supervised.gat.GATModule`).
Lightning tried to resume from `last.ckpt` and parsed the old class_path.
Fix: `rm -rf checkpoints/` in the 3 affected dirs. Dagster retries start fresh.

### Bug 7: DGI gets `variational` override it can't accept (set_01 ablation)

**Source:** `graphids/config/topology.py`, `graphids/orchestrate/planning.py`
**Commit:** `844cb23`

After fixing bug 5, DGI failed with: `Option 'variational' is not accepted`.
`variational` was added to autoencoder's `identity_keys` (for dedup) and the override
loop passed all identity keys as `--model.init_args.X` CLI overrides. DGI doesn't
have a `variational` param. Fix: add explicit `model_keys` to autoencoder stage def
(subset of identity_keys), and skip `variational` for DGI in the override loop.

### Bug 8: Conv-variant autoencoders use wrong model config (set_01 ablation)

**Source:** `graphids/orchestrate/planning.py`
**Commit:** `844cb23`

Claim 5 curriculum sweeps set `model_family: gat` with `conv_type: [gat, gps]`.
The `model_type: gat` propagated to upstream autoencoder stages, causing them to
load GAT config files (`gat/base.yaml`) instead of VGAE config files. Autoencoders
should always use VGAE (or DGI) — `conv_type` is a param within the VGAE model.
Fix: only override unsupervised model_family when `model_type` is actually an
unsupervised model (`_UNSUPERVISED_MODELS = {"vgae", "dgi"}`).

## Dagster Retry Behavior

`RetryPolicy(max_retries=2, delay=30)` — 3 total attempts per asset per run.
Once exhausted, dagster cannot retry within the same run. No external signal mechanism.
Recovery requires a new `dg launch` or a manual SLURM submission for leaf nodes.

The DGI autoencoder exhausted retries (bugs 5+7 burned all 3 attempts). Submitted
manually via standalone `sbatch` since it's a leaf node with no downstream.
