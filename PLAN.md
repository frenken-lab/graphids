# KD-GAT Session Plan

> Last updated: 2026-04-01 (session 6 — smoke test debug, decouple checks, register buffers)

## Current State

Pipeline converges at LightningCLI (`train_entrypoint.py` → `run_lightning()`). ConfigResolver
handles cross-field validation + audit trail. SLURM submission via `scripts/submit.sh` works.
Dagster orchestrator runs as CPU SLURM job (not login node).

Each model config is **one dagster asset = one SLURM job** running train→test→analyze
sequentially. Test/analyze are best-effort (`set +e`) so training success is preserved.

Asset checks are split: `checkpoint_complete` (blocking) gates downstream assets,
`analysis_complete` (non-blocking) is informational only.

## What this session did (2026-04-01, session 6)

### Smoke test (hcrl_sa, seed 99, 10 epochs)

Ran all 4 stages end-to-end: autoencoder→normal→curriculum→fusion. All 4 SLURM jobs
completed (0 errors). Dagster check failed only because check code was renamed mid-run
(deployment artifact, not a code bug). Clean resubmit would pass.

### 12 bugs found and fixed

**Config (4):** `_flatten_dict` double-prefix on qualified keys; curriculum epoch mismatch
(`data.init_args.max_epochs` not synced); `max_epochs` not fully qualified for YAML validation;
`FusionDataModule` rejecting curriculum-only `max_epochs` override.

**Device (3):** `_vgae_weights` CPU/CUDA mismatch; `alpha_values` (bandit) plain tensor not
buffer; `_alpha_values_t` (DQN) same pattern.

**Model (1):** `test_step` missing `dataloader_idx` in 4 models (7 definitions total).

**Pipeline (4):** `write_manifest` defined but never called; `outputs_complete` check too strict
(blocked downstream on analysis failure); `set -e` killing analyze on test failure; dagster
partition format.

### Architectural improvements

- `FusionRewardCalculator` → `nn.Module` with `register_buffer` (principled device management)
- Asset checks split: `checkpoint_complete` (blocking) + `analysis_complete` (non-blocking)
- Curriculum epoch auto-sync in ConfigResolver (recipes don't need stage-specific overrides)
- `generate_script` uses `set +e` before test/analyze (training success preserved)

### Issues filed

| Issue | Problem |
|-------|---------|
| `issues/per-stage-recipe-overrides.md` | Global `trainer_overrides` can't handle stage-specific params |
| `issues/slurm-phase-reporting.md` | Dagster can't distinguish partial job success |
| `issues/analyzer-manifest-lifecycle.md` | Manifest writer/checker split across 3 files |
| `issues/override-pipeline-consolidation.md` | 4-hop override flow prone to transform bugs |

## Blocking — Must do before ablation

1. **Run clean smoke test** — resubmit with current code (checks renamed, no mid-run conflict):
   ```bash
   KD_GAT_RECIPE=graphids/config/recipes/smoke_test.yaml \
     scripts/submit.sh ablation --assets '*' --partition 'hcrl_sa|99'
   ```

2. **Run config/override tests on SLURM**:
   ```bash
   scripts/submit.sh tests -k "test_overrides or test_config or test_merge_parity or test_submit_sh or test_cli_routing or test_recipe_expand_kd"
   ```

3. **Fusion checkpoint issue** — `best_model.ckpt` not created for bandit (RL training doesn't
   log the metric `ModelCheckpoint` monitors). Either configure checkpoint to save on epoch end
   or use `last.ckpt` as fallback in `test-from-spec`.

## Next

1. Clean smoke test (blocking item 1)
2. Run tests (blocking item 2)
3. Fix fusion checkpoint config
4. Launch ablation (`plans/experiment-sweep-plan.md`)

## Key References

| Doc | Purpose |
|-----|---------|
| `plans/architecture/slurm-job-consolidation.md` | **Implemented** — bundle train+test+analyze in one SLURM job |
| `issues/config-system-overhaul.md` | Config overhaul tracker — completed + open items |
| `issues/per-stage-recipe-overrides.md` | Global vs stage-specific overrides |
| `issues/slurm-phase-reporting.md` | Per-phase success reporting |
| `issues/analyzer-manifest-lifecycle.md` | Manifest ownership |
| `issues/override-pipeline-consolidation.md` | 4-hop override flow |
| `plans/experiment-sweep-plan.md` | 17-config ablation matrix |
| `plans/open_issues.md` | All deferred items |
