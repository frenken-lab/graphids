# Config System Overhaul

> Extracted from `plans/research/config_system_synthesis.md`
> Created: 2026-03-31

---

## Phase 1 — YAML restructuring + forced callbacks (no new abstractions)

### P0: Forced callbacks via `add_lightning_class_args`

- **Problem:** jsonargparse replaces lists atomically. Any stage YAML that defines
  `trainer.callbacks:` silently drops ModelCheckpoint + EarlyStopping from `trainer.yaml`.
  Already caused data loss — curriculum runs trained 300 epochs with no checkpoint.
- **Fix:** Register ModelCheckpoint and EarlyStopping via `parser.add_lightning_class_args()`
  in `GraphIDSCLI.add_arguments_to_parser()`. They get separate namespaces (`checkpoint.*`,
  `early_stopping.*`) immune to `trainer.callbacks:` list replacement. Stage YAMLs override
  via `checkpoint.monitor: val_acc`, not `trainer.callbacks: [...]`.
- **Files:** `graphids/cli.py`, `graphids/config/trainer.yaml`, `graphids/config/stages/fusion*.yaml`,
  `graphids/orchestrate/validate.py`
- **Plan:** `plans/architecture/forced-callbacks.md`
- **Risk:** None — strictly additive
- **Validation:** Spike one curriculum + one fusion run on `gpudebug`, confirm checkpoints save
- [x] Register ModelCheckpoint + EarlyStopping via `add_lightning_class_args` in `cli.py`
- [x] Set defaults (monitor, mode, save_top_k, patience) in `cli.py`
- [x] Remove ModelCheckpoint + EarlyStopping from `trainer.yaml` callbacks list
- [x] Replace `callbacks:` blocks in fusion stage YAMLs with namespace overrides
- [x] Simplify `before_instantiate_classes` — remove interim ModelCheckpoint guard
- [x] Add monitor metric validation in `orchestrate/validate.py`
- [ ] Spike: submit one curriculum + one fusion job on `gpudebug`
  - **Skipped** — requires SLURM (login node). Run before next experiment relaunch.

### P1: Separate cross-product overlays into independent axes

- **Problem:** Overlay files (`small_gat.yaml`, `large_vgae.yaml`) encode scale x model in a
  single file. File count grows quadratically: 3 models x 3 scales = 9 files instead of 6.
  Missing overlays (e.g., `large_dgi.yaml`) are silently skipped with no warning.
- **Fix:** Split into independent directories: `graphids/config/scales/` (one file per scale)
  and `graphids/config/models/` (one file per model type). Each axis is composable
  independently via multiple `--config` flags.
- **Files:** `graphids/config/overlays/` (split), `graphids/orchestrate/component.py`
  (`_resolve_config_files`), recipe YAMLs
- **Risk:** Low — file reorganization, no new code
- [ ] Audit current overlays — determine which params are scale-only vs model-only vs coupled
- [ ] Create `graphids/config/scales/{small,large}.yaml` with scale-only params
- [ ] Create `graphids/config/models/{vgae,gat,dgi,dqn}.yaml` with model-only params
- [ ] Update `_resolve_config_files()` in `component.py` to compose axis files independently
- [ ] Update recipe YAMLs to reference axis files
- [ ] Delete old cross-product overlay files
- [ ] Verify: `python -m graphids.orchestrate validate` passes

### P1: Import-time cross-validation of resources vs pipeline topology

- **Problem:** `resources.yaml` can drift from `pipeline.yaml`. Currently: `dgi/large` has no
  resource profile, `evaluation` stage has none, `medium` scale entries are dead config.
  Only caught at dagster `build_defs()` time — not at import or test time.
- **Fix:** Add an assertion in `graphids/config/__init__.py` (extending the existing
  `ckpt_stages` pattern) that cross-validates every `(model_type, scale, stage)` in
  `pipeline.yaml` has a corresponding entry in `resources.yaml`.
- **Files:** `graphids/config/__init__.py`, `graphids/config/resources.yaml` (clean dead entries)
- **Risk:** None — one assertion
- [ ] Add import-time assertion in `config/__init__.py`
- [ ] Remove dead `medium` scale entries from `resources.yaml` (or add `medium` to `pipeline.yaml`)
- [ ] Add `dgi/large` resource profile or remove `dgi` from large-scale pipeline
- [ ] Decide on `evaluation` stage: add resource profile or mark as not-yet-dagster-managed

---

## Phase 2 — Narrow typed contract (~150 lines, Dagster<->Lightning boundary)

### P2: `TrainingRunConfig` Pydantic schema contract

- **Problem:** Three config domains (model/trainer, orchestration, experiment) with no enforced
  contract. Pack/unpack impedance across the Dagster -> SLURM -> Lightning process boundary.
  Dagster can't validate training config — it trusts the YAML chain parses at runtime.
- **Fix:** A narrow Pydantic `TrainingRunConfig` (10-20 parameters that cross the boundary or
  are actively swept). `to_lightning_yaml()` / `from_lightning_yaml()` as the single
  serialization boundary. `extra="forbid"` catches drift.
- **Scope discipline:** Only parameters that cross Dagster<->Lightning or are actively swept.
  Internal Lightning details stay in YAML.
- **Risk:** Medium — new abstraction, scope discipline required to prevent bloat
- [ ] Define `TrainingRunConfig` with boundary/swept params only
- [ ] Implement `to_lightning_yaml()` and `from_lightning_yaml()`
- [ ] Write unit tests for round-trip serialization

### P2: `ConfigResolver` as exclusive pipeline merge path

- **Problem:** Override resolution is implicit — whatever jsonargparse merges last wins, with
  no audit trail. Cross-field constraint violations can pass individual-layer validation.
- **Fix:** `ConfigResolver.resolve()` as the exclusive merge path for pipeline runs. Validates
  final merged state. Emits override audit log (which override came from which source).
  jsonargparse native merge used only for dev/test CLI invocations.
- **Risk:** Medium — replaces existing merge path for dagster
- [ ] Implement `ConfigResolver` with explicit priority order
- [ ] Wire into dagster `component.py` asset creation
- [ ] Add override audit logging

### P2: Replace `write_paths.yaml` with frozen `PathContext`

- **Problem:** `write_paths.yaml` declares paths but nothing enforces them. `run_dir()` in
  `config/__init__.py` duplicates the pattern as an f-string. Code can write anywhere.
- **Fix:** Frozen Pydantic `PathContext` with computed properties is the only source of write
  paths. Inject into LightningCLI callbacks and Dagster ops. Delete `write_paths.yaml`.
- **Risk:** Medium — touches all write sites
- [ ] Implement `PathContext` (frozen, computed properties)
- [ ] Wire as `ConfigurableResource` on Dagster side
- [ ] Update `GraphIDSCLI` to use `PathContext` for logger/checkpoint dirs
- [ ] Enforce the central invariant: resolved config written to `config_snapshot_path` before training
- [ ] Delete `write_paths.yaml` and remove `run_dir()` f-string duplicate

---

## Phase 3 — Ongoing discipline + optional enhancements

### P3: Scope discipline for `TrainingRunConfig`

- **Problem:** Risk of `TrainingRunConfig` growing to mirror every `__init__` parameter.
- **Mitigation:** Treat any field addition as a deliberate decision. `extra="forbid"` catches
  accidental additions at definition time.
- [ ] Document the boundary rule in CLAUDE.md or rules/
- [ ] Review `TrainingRunConfig` fields quarterly — remove any that aren't actively used

### P3: Recipe generation as code

- **Problem:** Recipe YAMLs don't scale with ablation dimensions — each new dimension
  multiplies entries.
- **Fix:** Python functions that generate recipe configs parametrically (Pattern 3 applied
  narrowly to sweep enumeration only). Generates YAMLs that LightningCLI reads — no runtime
  coupling.
- **Risk:** Low — isolated to `orchestrate/`
- [ ] Evaluate whether current recipe complexity justifies this
- [ ] If yes, implement recipe generator in `orchestrate/`
