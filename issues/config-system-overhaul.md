# Config System Overhaul

> Canonical design: `plans/research/config_system_synthesis.md`
> Created: 2026-03-31 | Updated: 2026-04-01

---

## Completed

<details>
<summary>Phase 1 — YAML restructuring + forced callbacks ✓</summary>

- **P0: Forced callbacks** — `add_lightning_class_args(ModelCheckpoint/EarlyStopping)` in
  `cli.py:39-52`. Separate namespaces immune to list replacement. Fixed data loss bug.
- **P1: Cross-product split** — `config/models/{type}/{base,scales/{small,large}}.yaml`.
  Old cross-product overlays deleted. Linear file growth.
- **P1: Import-time cross-validation** — `topology.py:90-114` validates model configs,
  fusion configs, and resource profiles at import. 5 tests.

</details>

<details>
<summary>Phase 2.1 — TrainingRunConfig schema contract ✓</summary>

- Narrow Pydantic model (8 fields, `extra="forbid"`, frozen) in `config/contracts.py`.
- `KDEntry` sub-schema. Validates against topology constants at construction.
- `enumerate_assets()` migrated from untyped dicts. 20 tests.

</details>

<details>
<summary>Phase 2.2 — ConfigResolver (with YAML-aware validation) ✓</summary>

- `orchestrate/resolve.py`: single merge point replacing two separate sites.
- Subsumes `training_spec()` (deleted from `execution.py`) and `apply_resource_overrides()`
  call (removed from `assets.py`).
- `ResolvedConfig` carries `TrainingSpec` + `ResourceSpec` + paths + audit trail.
- `OverrideRecord` tracks key, value, source for every override.
- Override-layer validators: `num_workers ≤ cpus_per_task-1`, dead `batch_size` override
  for RL fusion, GPU partition consistency.
- YAML-aware validators via `_merge_yaml_chain()`: curriculum epoch sync
  (`data.init_args.max_epochs` vs `trainer.max_epochs`), YAML `num_workers` vs resource
  profile CPUs, RL dead `batch_size` in stage YAML.
- Structured audit logging via structlog.
- Tests: 30 total (5 deep merge, 3 dotted overrides, 5 YAML-aware validation,
  3 audit, 3 cross-field, 11 existing override/recipe tests).

</details>

---

## Open — Ordered by Priority

### 1. SLURM validation (blocks everything)

Run tests and smoke test on SLURM to validate all recent changes.

```bash
scripts/submit.sh tests -k test_overrides
scripts/submit.sh tests -k test_config
KD_GAT_RECIPE=graphids/config/recipes/smoke_test.yaml dg launch --assets '*'
```

- [ ] Override tests pass on SLURM
- [ ] Config tests pass on SLURM
- [ ] Smoke test completes (autoencoder → curriculum → fusion)

### ~~2. W1: Cross-field validators don't read YAML configs~~ ✓

Resolved 2026-04-01. Added `_merge_yaml_chain()` (naive deep merge + dotted-key
override application) and 3 YAML-aware validators: curriculum epoch sync, YAML
num_workers vs CPUs, RL dead batch_size. 13 new tests. See `resolve.py`.

### ~~3. W2: Structural exclusivity~~ ✓

Resolved 2026-04-01. Decoupled CLI architecture eliminates dual merge:
- `artifact_paths()` deleted — `PathContext` is single source for paths.
- `checks.py` uses `PathContext` directly.
- Pipeline path uses direct instantiation (`train_entrypoint.py`), no LightningCLI.
- `to_cli_overrides()` replaced by `to_override_dict()` (dict, not CLI strings).
- `StageConfig` override fields still consumed exclusively by `ConfigResolver`.
- See `plans/research/config_cli_decoupling.md`.

### ~~4. P2.3: Replace `run_dir()` with frozen `PathContext`~~ ✓

Resolved 2026-04-01. `PathContext` (frozen Pydantic model) in `config/paths.py`.
Properties: `run_dir`, `ckpt_file`, `complete_marker`, `last_ckpt_file`, `ckpt_dir`.
`run_dir()` function deleted. `ResolvedConfig.paths: PathContext` replaces three
separate path fields.

### 5. W3: Per-stage override granularity (MEDIUM)

`trainer_overrides` and `resource_overrides` apply uniformly to all stages.
"autoencoder gets 2 epochs, curriculum gets 5" is not expressible. Limitation
is in `planning.py` (recipe-level fields assigned to every `StageConfig`).

- [ ] Evaluate whether current recipes need per-stage granularity
- [ ] If yes: per-stage override blocks in recipe envelope, different values per `StageConfig`

### 6. P2.4: Align `_KDSpec` / `KDEntry` field sets (LOW-MEDIUM)

`_KDSpec` (recipe_expand.py) has 7 fields; `KDEntry` (contracts.py) has 3. The 4
extra fields (`temperature`, `model_path`, `vgae_latent_weight`, `vgae_recon_weight`)
bypass `TrainingRunConfig.auxiliaries` validation.

- [ ] Decide: identity-relevant (add to `KDEntry`) or sweep-internal (document split)

### 7. W4: Override collision detection (LOW)

Last-write-wins if `trainer_overrides` and `model_init_overrides` touch the same key.
Audit log records what was applied but doesn't warn on overwrites.
- [ ] Warn if a key appears in multiple override sources

### 8. W5: `OverrideRecord.value` type is lossy (LOW)

KD overrides audit stores JSON blob as string. Not queryable.
- [ ] Widen to `str | int | float | dict | list` or store structured + rendered

### 9. W7: Spec-file path bypasses validators (LOW)

`train-from-spec` now uses `resolve_configs()` to merge the YAML chain before
direct instantiation, so the same merge path runs on both dagster and SLURM sides.
Cross-field validation still only runs in `ConfigResolver` (dagster side).
`analyze-from-spec` is unchanged.
- [ ] Wire cross-field validators into `train-from-spec` path (optional)

### 10. W6: Resume checkpoint probe is side effect (LOW)

`resume.exists()` makes resolver output depend on filesystem state. Correct
behavior, but requires filesystem fixtures for unit tests.
- [ ] Accept as inherently stateful, or extract to a callback

---

## Phase 3 — Ongoing discipline

### Scope discipline for `TrainingRunConfig`

`extra="forbid"` enforced. Any field addition is a deliberate decision.
- [ ] Document the boundary rule in `rules/`
- [ ] Review fields quarterly — remove unused

### Recipe generation as code

Recipe YAMLs don't scale with ablation dimensions. Python functions could generate
configs parametrically. Low priority — evaluate when recipe complexity demands it.
- [ ] Evaluate whether current complexity justifies this

---

## Audit Log

### 2026-04-01

- Audited both docs against codebase. Phase 1 complete, P2.1 complete.
- Interim field-passthrough override flow superseded by ConfigResolver implementation.
- Deleted `training_spec()` from `execution.py` (subsumed by resolver).
- Found `_KDSpec`/`KDEntry` field divergence (7 vs 3) — added as P2.4.
- Identified 7 weaknesses in ConfigResolver, documented with severity and fix paths.
- Decision: ConfigResolver is target architecture, current implementation is v1.
- Fixed fusion `vgae_weights` bug — config gap in method YAMLs, deleted `set_vgae_weights()`.
- Added YAML-aware validation (`_merge_yaml_chain` + 3 validators + 13 tests). W1 resolved.
- Researched Dynaconf (rejected) and OmegaConf (not recommended) — `config_tool_comparison.md`.
- Documented Phase 3 CLI decoupling architecture — `config_cli_decoupling.md`.

### 2026-04-01 (session 2)

- Built decoupled CLI architecture (Phase 3 from `config_cli_decoupling.md`).
- `cli.py` is now torch-free — `resolve_configs()` + `run_lightning()` (lazy import).
- LightningCLI subclass moved to `_lightning.py` (internal, dev path + validate only).
- Pipeline path uses direct instantiation (`train_entrypoint.py`) — no LightningCLI.
- `PathContext` (frozen Pydantic) replaces `run_dir()` + `artifact_paths()`.
- `ResolvedConfig.paths: PathContext` replaces 3 separate path fields (no compat layer).
- `to_cli_overrides()` → `to_override_dict()` (dict output, not CLI strings).
- Merge functions (`deep_merge`, `apply_dotted_overrides`, `merge_yaml_chain`) moved to `yaml_utils.py`.
- `validate.py` uses snapshot path instead of reconstructing CLI args.
- `__main__.py` defers `torch.multiprocessing` setup to `_run_lightning()`.
- W2 resolved, P2.3 resolved, W7 partially resolved.

---

## Reference

- `plans/research/config_system_synthesis.md` — canonical design (Parts 1-7)
- `graphids/config/CONFIG_REFERENCE.md` — parameter axes and infrastructure
- `~/.claude/plans/purrfect-twirling-diffie.md` — P2.1 implementation plan
