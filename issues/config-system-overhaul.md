# Config System Overhaul

> Extracted from `plans/research/config_system_synthesis.md`
> Created: 2026-03-31
> Updated: 2026-03-31 — Phase 1 complete, P2.1 complete

---

## Phase 1 — YAML restructuring + forced callbacks (no new abstractions) ✓

### P0: Forced callbacks via `add_lightning_class_args` ✓

- **Problem:** jsonargparse replaces lists atomically. Any stage YAML that defines
  `trainer.callbacks:` silently drops ModelCheckpoint + EarlyStopping from `trainer.yaml`.
  Already caused data loss — curriculum runs trained 300 epochs with no checkpoint.
- **Fix:** Register ModelCheckpoint and EarlyStopping via `parser.add_lightning_class_args()`
  in `GraphIDSCLI.add_arguments_to_parser()`. They get separate namespaces (`checkpoint.*`,
  `early_stopping.*`) immune to `trainer.callbacks:` list replacement.
- **Files:** `graphids/cli.py`, `graphids/config/trainer.yaml`, `graphids/config/stages/fusion*.yaml`,
  `graphids/orchestrate/validate.py`
- [x] All implementation subtasks complete
- [ ] Spike: submit one curriculum + one fusion job on `gpudebug` (pre-relaunch)

### P1: Separate cross-product overlays into independent axes ✓

- **Problem:** Overlay files (`small_gat.yaml`, `large_vgae.yaml`) encode scale × model in a
  single file. File count grows quadratically. Missing overlays silently skipped.
- **Fix:** `graphids/config/models/{model_type}/{scale}.yaml` — one directory per model type,
  one file per scale. KD overlays at `models/{model}/kd.yaml`. Old cross-product overlays deleted.
- **Files:** `graphids/config/models/` (7 model dirs, 16 files), `graphids/config/overlays/`
  (reduced to `profile.yaml`), `graphids/orchestrate/component.py`
- [x] All implementation subtasks complete
- [x] Bandit scale files populated (hidden_dim, num_layers, backbone_retrain_freq, backbone_epochs)
- [x] Bandit shared defaults surfaced in `fusion_bandit.yaml` (ucb_alpha, lambda_reg, backbone_lr)
- [x] Bandit resource profiles bumped for backbone retraining overhead
- [x] Backbone eval mode leak fixed in `bandit.py:retrain_backbone()`

### P1: Import-time cross-validation of resources vs pipeline topology ✓

- **Problem:** `resources.yaml` drifts from `pipeline.yaml`. Dead config, missing profiles,
  unreachable entries — only caught at dagster `build_defs()` time.
- **Fix:** Three-way cross-validation at import time: `pipeline.yaml` × `models/` × `resources.yaml`.
  `resource_model` field on `StageConfig` so fusion assets use method-specific resource profiles.
- **Files:** `graphids/config/__init__.py`, `graphids/config/pipeline.yaml`,
  `graphids/config/resources.yaml`, `graphids/orchestrate/component.py`, `tests/config/test_config.py`
- [x] All implementation subtasks complete
- [x] 5 cross-validation tests in `tests/config/test_config.py`

**Phase 1 validation:** `python -m graphids.orchestrate validate` needs SLURM — run before relaunch.

---

## Phase 2 — Narrow typed contract (Dagster<->Lightning boundary)

### P2.1: `TrainingRunConfig` Pydantic schema contract ✓

- **Problem:** Recipe entries are untyped dicts. `enumerate_assets()` accesses them with string
  `.get()` calls — typos like `scael` or `conv_typ` are silently ignored and produce wrong
  checkpoint paths at SLURM submission time, not at recipe-load time.
- **Fix:** Narrow Pydantic `TrainingRunConfig` (8 fields that cross the boundary or are swept)
  with `extra="forbid"`. Validates against `pipeline.yaml` constants at construction.
  `.merge()` validates overlays. `KDEntry` sub-schema for KD auxiliary config.
- **Scope:** `stages`, `scale`, `conv_type`, `loss_fn`, `fusion_method`, `variational`,
  `model_type`, `auxiliaries`. Internal Lightning params stay in YAML.
- **Files:** `graphids/config/__init__.py` (+95 lines), `graphids/orchestrate/component.py`
  (migrated `enumerate_assets` + 3 helpers), `graphids/orchestrate/validate.py` (early schema
  validation), `tests/config/test_config.py` (+20 tests)
- [x] `KDEntry` + `TrainingRunConfig` schemas in `config/__init__.py`
- [x] `enumerate_assets()` migrated from untyped dict to `TrainingRunConfig`
- [x] `_overlay_model`, `_resolve_config_files`, `_identity_value` take typed `TrainingRunConfig`
- [x] Early schema validation in `validate_recipe()` — recipe errors before CLI parse errors
- [x] 20 tests: defaults, frozen, extra=forbid, validators, coercion, recipe round-trip
- [x] 80 tests pass (login node). SLURM validation pending.

### P2.2: `ConfigResolver` as exclusive pipeline merge path

- **Problem:** Override resolution is implicit — whatever jsonargparse merges last wins, with
  no audit trail. Cross-field constraints not validated:
  - `CurriculumDataModule.max_epochs` must match `trainer.max_epochs`
  - `num_workers` in stage YAML should be ≤ `cpus_per_task - 1` in resource profile
  - `FusionDataModule.batch_size` is dead when method is bandit/dqn
- **Fix:** `ConfigResolver.resolve()` as the exclusive merge path for pipeline runs. Validates
  final merged state. Emits override audit log (which override came from which source).
- **Risk:** Medium — replaces existing merge path for dagster
- [ ] Implement `ConfigResolver` with cross-field validators
- [ ] Wire into dagster `component.py` asset creation
- [ ] Add override audit logging

### P2.3: Replace `write_paths.yaml` with frozen `PathContext`

- **Problem:** `write_paths.yaml` declares paths but nothing enforces them. `run_dir()` in
  `config/__init__.py` duplicates the pattern as an f-string. Code can write anywhere.
- **Fix:** Frozen Pydantic `PathContext` with computed properties is the only source of write
  paths. Inject into LightningCLI callbacks and Dagster ops. Delete `write_paths.yaml`.
- **Risk:** Medium — touches all write sites. Lower priority than P2.2.
- [ ] Implement `PathContext` (frozen, computed properties)
- [ ] Wire as `ConfigurableResource` on Dagster side
- [ ] Update `GraphIDSCLI` to use `PathContext` for logger/checkpoint dirs
- [ ] Delete `write_paths.yaml` and remove `run_dir()` f-string duplicate

---

## Phase 3 — Ongoing discipline + optional enhancements

### P3: Scope discipline for `TrainingRunConfig`

- **Problem:** Risk of `TrainingRunConfig` growing to mirror every `__init__` parameter.
- **Mitigation:** `extra="forbid"` catches accidental additions. Treat any field addition as
  a deliberate decision.
- [x] `extra="forbid"` enforced at schema level
- [ ] Document the boundary rule in rules/
- [ ] Review `TrainingRunConfig` fields quarterly — remove any that aren't actively used

### P3: Recipe generation as code

- **Problem:** Recipe YAMLs don't scale with ablation dimensions — each new dimension
  multiplies entries.
- **Fix:** Python functions that generate recipe configs parametrically. Generates YAMLs
  that LightningCLI reads — no runtime coupling.
- **Risk:** Low — isolated to `orchestrate/`
- [ ] Evaluate whether current recipe complexity justifies this

---

## Reference

- Config audit: `graphids/config/CONFIG_REFERENCE.md` (merged from PARAMETER_AXES + INFRASTRUCTURE_REFERENCE)
- P2.1 plan: `~/.claude/plans/purrfect-twirling-diffie.md`
