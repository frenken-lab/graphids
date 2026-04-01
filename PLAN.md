# KD-GAT Session Plan

> Last updated: 2026-04-01 (session 3 — shadow instantiation deleted, audit)

## Current State

Pipeline path and dev path converge at LightningCLI. `train_entrypoint.py` builds CLI
args from `TrainingSpec` and delegates to `run_lightning()` — no shadow instantiation.
ConfigResolver handles cross-field validation + audit trail (not merge-for-instantiation).
Shared wiring constants (`LINK_TARGETS`, callback defaults) in `cli.py`, consumed by
`_lightning.py`. Ready to run smoke test on SLURM.

**Smoke test command:**
```bash
KD_GAT_RECIPE=graphids/config/recipes/smoke_test.yaml dg launch --assets '*'
```

## What this session did (2026-04-01, session 3)

### Audit: hand-rolled code in recent additions

Audited 833 lines added over 4 commits. Found `train_entrypoint.py` reimplemented
~80 lines of LightningCLI: `_import_class`, `_get_dotted`, `_set_dotted`, `_apply_links`
(link_arguments), `_patch_paths` (before_instantiate_classes), manual Model/Data/Trainer
instantiation. All deleted.

### Shadow instantiation → LightningCLI delegation

`run_training_from_spec` now builds CLI args and calls `run_lightning()`. Both paths
converge at LightningCLI. 144 → 51 lines. Zero drift risk.

### Shared wiring constants

Extracted `LINK_TARGETS`, `CHECKPOINT_DEFAULTS`, `EARLY_STOPPING_DEFAULTS` to `cli.py`
as single source of truth. `_lightning.py` imports them.

### Doc alignment

Updated `issues/config-system-overhaul.md` (W2, W7, audit log),
`plans/research/config_cli_decoupling.md` (Resolution section), PLAN.md.

### MLOps evaluation + fixes (C1, C2)

Independent evaluation against Hydra, MMEngine, Dagster, LightningCLI (vanilla).
Report: `plans/architecture/cli-config-evaluation.md`. Two critical fixes applied:
- C1: `tests/config/test_merge_parity.py` — 3 tests assert naive merge matches
  jsonargparse for representative config chains (autoencoder, normal, fusion).
- C2: `write_yaml()` now atomic (temp file + fsync + rename) for NFS safety.

## Blocking — Must do before ablation

1. **Run smoke test on SLURM** — validates pipeline with LightningCLI delegation path
2. **Run override + config tests on SLURM** — `scripts/submit.sh tests -k test_overrides`

## Next

1. Run smoke test on SLURM
2. Run tests on SLURM
3. Launch ablation (`plans/experiment-sweep-plan.md`)
4. Evaluation + analysis as dagster assets (`plans/architecture/evaluation-analysis-assets.md`)

## Key References

| Doc | Purpose |
|-----|---------|
| `issues/config-system-overhaul.md` | Config overhaul tracker — completed + open items |
| `plans/research/config_system_synthesis.md` | Config system design — canonical reference |
| `plans/research/config_cli_decoupling.md` | Phase 3 CLI architecture |
| `plans/research/config_tool_comparison.md` | Dynaconf/OmegaConf evaluation |
| `plans/experiment-sweep-plan.md` | 17-config ablation matrix |
| `plans/open_issues.md` | All deferred items |
