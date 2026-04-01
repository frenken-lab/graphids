# KD-GAT Session Plan

> Last updated: 2026-04-01 (recipe override flow + slurm extraction session)

## Current State

Recipe-level trainer and resource overrides are wired end-to-end. Smoke test recipe updated to use hcrl_sa + curriculum + gpudebug. SLURM infrastructure extracted to `graphids/slurm/`. Ready to run smoke test on SLURM.

**TEMPORARY:** `defaults/trainer.yaml` has `max_epochs: 3`. Revert to `300` before ablation launch.

**Smoke test command:**
```bash
KD_GAT_DRY_RUN=1 KD_GAT_RECIPE=graphids/config/recipes/smoke_test.yaml dg launch --assets '*'
```

**Real smoke test (on SLURM):**
```bash
KD_GAT_RECIPE=graphids/config/recipes/smoke_test.yaml dg launch --assets '*'
```

## Handoff — What this session did (2026-04-01)

### Problem analysis

Last session spent hours on a 3-epoch validation run, fixing 5 runtime bugs one-at-a-time through serial SLURM submit→fail→fix→resubmit cycles. Each cycle cost 5-15 min of queue wait. Root cause: no way to run a fast smoke test with short walltimes, and no validation on SLURM resource formats.

### Recipe override flow (Phase 2 partial)

Implemented the override resolution layer from `plans/research/config_system_synthesis.md` Part 5. Trainer and resource overrides flow through a separate channel from identity (TrainingRunConfig stays narrow at 8 fields).

```
recipe YAML
  ├── overrides       → defaults → TrainingRunConfig  (identity — unchanged)
  ├── trainer_overrides → flatten("trainer.") → StageConfig → runtime_overrides → CLI args
  └── resource_overrides → StageConfig → ResourceSpec patching → SLURM sbatch
```

Files changed:
- `graphids/config/recipe_expand.py` — `_RecipeEnvelope` gains `trainer_overrides` + `resource_overrides`, `_flatten_dict()` helper, passthrough in output
- `graphids/orchestrate/planning.py` — `StageConfig` gains `trainer_overrides: dict[str, str]` + `resource_overrides: dict[str, str | int]`, populated from recipe
- `graphids/orchestrate/execution.py` — `training_spec()` merges `trainer_overrides` into `runtime_overrides` before KD/resume
- `graphids/slurm/resources.py` — `apply_resource_overrides()` with key validation
- `graphids/orchestrate/assets.py` — resource overrides applied after `get_resources()`, before `scale_resources()`

### Smoke test recipe

`graphids/config/recipes/smoke_test.yaml` rewritten:
- `set_01` → `hcrl_sa` (3,995 graphs, 76 MB — fastest real dataset)
- Added `curriculum` stage (exercises VGAE checkpoint seam via `CurriculumDataModule.setup()` → `load_inner_model("vgae", ckpt)`)
- `trainer_overrides: {max_epochs: 2}` — cold + warm DataLoader start
- `resource_overrides: {time: "0:15:00", partition: gpudebug}` — priority queue, fast turnaround

### SLURM time format validation

`ResourceSpec.__post_init__` validates time format matches `[D-]HH:MM:SS`. Catches the exact `2:00` (interpreted as 2min) bug from last session at construction time — applies to all sources: profile YAMLs, resource_overrides, scale_resources output.

`submit_profile.py` also routes through `ResourceSpec` for validation, covering the `scripts/submit.sh` submission path.

### SLURM package extraction

Extracted `resources.py` and `slurm.py` from `graphids/orchestrate/` to `graphids/slurm/`. These are infrastructure (no dagster imports) that was buried under orchestration. Two commands already reached in from outside (`commands/profile.py`, `commands/submit_profile.py`).

New import hierarchy:
```
graphids/config/   (top — no deps on other graphids packages)
graphids/core/     (models, preprocessing, contracts)
graphids/slurm/    (ResourceSpec, submit, sacct_query — SLURM infra)
graphids/orchestrate/  (dagster-specific: assets, component, planning)
```

### Tests

`tests/orchestrate/test_overrides.py` — 17 tests covering:
- `_flatten_dict` (nested→dotted conversion, booleans, edge cases)
- Recipe expansion (trainer flattening, resource passthrough, backward compat)
- `apply_resource_overrides` (field patching, typo rejection, noop on empty)
- Trainer override flow (StageConfig → runtime_overrides → CLI args)

### Checkpoint flow documented

Traced the inter-stage checkpoint loading:
- `CurriculumDataModule.setup()` loads VGAE checkpoint for difficulty scoring
- `FusionDataModule.setup()` loads VGAE + GAT checkpoints for state vector caching
- Both use `load_inner_model()` → `cls.load_from_checkpoint()`
- `fast_dev_run=True` disables ModelCheckpoint — would break downstream stages. Must use `max_epochs` + `limit_*_batches` for smoke tests.

## Blocking — Must fix before ablation relaunch

1. **Revert `max_epochs`** — `defaults/trainer.yaml` is at 3 for validation. Set back to 300.
2. **Run smoke test on SLURM** — validate the full pipeline end-to-end with the new recipe.
3. **Fusion vgae_weights init** — pre-existing bug, blocks fusion stage. Not addressed this session.
4. **Recipe `overrides` for trainer params** — RESOLVED by `trainer_overrides` field.
5. **Orchestrate test coverage** — `test_overrides.py` added. Integration tests (`test_dagster_integration.py`) exist but may have stale imports.

## In Progress

- HF Spaces dashboard (`buckeyeguy/kd-gat-dashboard`)

## Next (after smoke test succeeds)

1. Run full smoke test on SLURM — `KD_GAT_RECIPE=.../smoke_test.yaml dg launch --assets '*'`
2. Fix fusion vgae_weights init bug
3. Revert max_epochs to 300, launch ablation
4. Evaluation + analysis as dagster assets (`plans/architecture/evaluation-analysis-assets.md`)
5. Phase 2 remaining: `PathContext`, env var wiring (`KD_GAT_SLURM_PARTITION`)

## Key References

| Doc | Purpose |
|-----|---------|
| `plans/research/config_system_synthesis.md` | Config system design — override resolution, Phase 1-3 |
| `plans/open_issues.md` | All deferred items, consolidated |
| `plans/experiment-sweep-plan.md` | 17-config ablation matrix, stage-sharing DAG, phased HPO |
| `plans/ablation_and_main_005.md` | Run 005 job summary + failure post-mortem |
| `.claude/plans/swirling-watching-codd.md` | Implementation plan for this session's work |
