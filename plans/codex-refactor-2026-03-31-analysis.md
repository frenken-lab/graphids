# Codex Refactor Analysis — 2026-03-31

> Post-refactor audit of commits `57fa1b8..0407ddc` (5 commits, 138 files, +5485/−2391).
> Source doc: `graphids/REFACTOR_NOTES_2026-03-31.md`.

## TL;DR

The refactor achieved its structural goals: config modularized, orchestrate decomposed,
models reorganized into family namespaces, contracts formalized. But it was a **static-only
refactor** — no runtime or test execution happened. As a result:

- **~25 test imports are broken** (old model paths, deleted orchestrate symbols)
  USER NOTEL: Deleted tests with IOManager so now no issue :)
- **Dataset catalog is broken** (`CATALOG_PATH` → deleted file, crashes dagster + datamodule)
- **`CheckpointPathIOManager` deleted** but 4 test files still import it
- **6+ new YAML files are orphaned** (schema/, overrides/, recipes) — no Python loads them
- **Analysis path runs in-process** in dagster worker, not via SLURM — likely unintentional
- **Documentation is stale** (CLAUDE.md, config-system.md rules still describe old structure)

---

## 1. What Works (Wired End-to-End)

### Config facade

`config/__init__.py` is now a clean re-export facade. Six submodules (`base`, `runtime`,
`topology`, `paths`, `contracts`, `yaml_utils`) + `recipe_expand` are all imported and have
real consumers. No circular imports. Import order is clean:
`base → yaml_utils → topology → runtime → paths → contracts → recipe_expand`.

### Training contract chain (dagster → SLURM → worker)

The full chain connects:

```
definitions.py → component.build_defs()
  → planning.enumerate_assets() → StageConfig list
  → assets.make_training_asset() → @dg.asset
    → execution.training_spec() → TrainingSpec
    → component.SlurmTrainingResource.submit_and_wait()
      → slurm.SubprocessSlurmJobClient.run_training_job()
        → writes spec.json (TrainingContract.to_envelope)
        → sbatch: "python -m graphids train-from-spec --spec-file ..."
        → polls to completion

[On SLURM worker:]
  __main__.py → train_from_spec.main()
    → _spec_payload.load_payload() → raw dict
    → train_entrypoint.run_training_from_payload()
      → TrainingContract.from_envelope() → TrainingSpec
      → GraphIDSCLI(args=["fit", "--config", ...])
```

### CLI entry point

`__main__.py` uses explicit `_COMMAND_MODULES` dict. Both `train-from-spec` and
`analyze-from-spec` are registered with real command files. `_build_parser()` creates
argparse subparsers for all commands.

### Model reorg

Four family directories exist with correct files:

- `autoencoder/`: `vgae.py`, `dgi.py`
- `supervised/`: `gat.py`
- `fusion/`: `bandit.py`, `dqn.py`, `fusion_baselines.py`, `fusion_features.py`, `fusion_reward.py`
- `temporal_family/`: `temporal.py`

`_MODULE_PATHS` in `_training.py` updated to new paths. Stage YAMLs use short `class_path`
names (e.g., `VGAEModule`), resolved by jsonargparse `subclass_mode_model=True` — no
import path mismatch in YAMLs.

### Config tree validation

`topology.py` validates at import time: every `(model_type, scale)` pair must have
`models/{family}/base.yaml` + `models/{family}/scales/{scale}.yaml`, every fusion method
must have `fusion/methods/{method}.yaml`, and every family must have a resource profile.

---

## 2. What's Broken

### 2.1 Dataset catalog — WILL CRASH at runtime

**Severity: CRITICAL**

`paths.py:14` sets `CATALOG_PATH = CONFIG_DIR / "datasets.yaml"`. That file was deleted.
The guard at `paths.py:15` (`if CATALOG_PATH.exists()`) makes `DEFAULT_DATASET` fall back
silently to `"set_01"`, so config import doesn't crash.

But two callers use `CATALOG_PATH` without a guard:

- `component.py:19` — `from graphids.config import CATALOG_PATH` → `read_yaml(CATALOG_PATH)`
  at `build_defs()` time. **Crashes all dagster materialization.**
- `datamodule.py` — `CATALOG_PATH.read_text()`. **Crashes at training runtime.**

The replacement (`config/datasets/*.yaml`, one file per dataset) was created but no Python
code iterates that directory. The catalog is effectively gone with no working replacement.

**Action needed:** Either restore `datasets.yaml` or wire up the per-file dataset configs.

### 2.2 Test suite — ~25 broken imports

**Severity: HIGH** (blocks all test execution)

#### Model import paths (old flat → new family namespaces)

| Test file                            | Broken import                                 | Should be                   |
| ------------------------------------ | --------------------------------------------- | --------------------------- |
| `tests/core/models/test_vgae.py`     | `graphids.core.models.vgae`                   | `.autoencoder.vgae`         |
| `tests/core/models/test_gat.py`      | `graphids.core.models.gat`                    | `.supervised.gat`           |
| `tests/core/models/test_temporal.py` | `graphids.core.models.temporal`               | `.temporal_family.temporal` |
| `tests/core/models/test_fusion.py`   | `graphids.core.models.fusion_features`        | `.fusion.fusion_features`   |
| `tests/core/models/test_fusion.py`   | `graphids.core.models.fusion_baselines`       | `.fusion.fusion_baselines`  |
| `tests/core/models/test_fusion.py`   | `graphids.core.models.fusion_reward`          | `.fusion.fusion_reward`     |
| `tests/core/models/test_fusion.py`   | `graphids.core.models.dqn`                    | `.fusion.dqn`               |
| `tests/test_smoke.py`                | `graphids.core.models.gat`, `.vgae`           | new paths                   |
| `tests/test_integration.py`          | `graphids.core.models.gat`, `.dqn`, `.bandit` | new paths                   |

Note: `models/__init__.py` does NOT re-export `VGAEModule`, `GATModule`, etc. by short
name — only fusion helpers and `GraphModuleBase`. So `from graphids.core.models.vgae import
VGAEModule` fails; must use `from graphids.core.models.autoencoder.vgae import VGAEModule`.

#### Orchestrate imports (symbols moved out of `component.py`)

| Test file                                       | Broken symbol                                                                      | Now lives in                              |
| ----------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------- |
| `tests/orchestrate/test_pure.py`                | `StageConfig`, `_cli_val`, `_identity_value`, `build_cli_args`, `enumerate_assets` | `orchestrate/planning.py`                 |
| `tests/orchestrate/test_dagster_unit.py`        | `CheckpointPathIOManager`, `_make_asset`, `build_cli_args`                         | **deleted** / `assets.py` / `planning.py` |
| `tests/orchestrate/test_dagster_integration.py` | `CheckpointPathIOManager`, `_make_asset`, `_make_checkpoint_checks`                | **deleted** / `assets.py` / `checks.py`   |
| `tests/orchestrate/conftest.py`                 | `CheckpointPathIOManager`, `StageConfig`                                           | **deleted** / `planning.py`               |
| `tests/orchestrate/test_iomanager.py`           | `CheckpointPathIOManager`                                                          | **deleted**                               |
| `tests/config/test_config.py:141`               | `enumerate_assets` from `component`                                                | `planning.py`                             |

#### Mock patches targeting wrong namespaces

`test_dagster_unit.py` and `test_dagster_integration.py` mock
`graphids.orchestrate.component.submit`, `.poll`, `.sacct_query`, `.generate_script` —
these functions live in `orchestrate/slurm.py`, not `component.py`. The mocks will silently
no-op (patching a namespace that doesn't contain those names).

### 2.3 `CheckpointPathIOManager` — deleted, not replaced in tests

USER NOTE: Deleted Tests so no issue

---

## 3. Orphaned Files (Added but Never Loaded)

| File                                      | Purpose (apparent)                              | Status                                   |
| ----------------------------------------- | ----------------------------------------------- | ---------------------------------------- |
| `config/matrix/allowed_combinations.yaml` | Allow/deny rules for model×stage combos         | No Python reads it                       |
| `config/schema/run.schema.yaml`           | JSON Schema for run requests                    | No `jsonschema.validate()` call          |
| `config/schema/model.schema.yaml`         | JSON Schema for model configs                   | Same                                     |
| `config/schema/fusion_method.schema.yaml` | JSON Schema for fusion methods                  | Same                                     |
| `config/schema/resources.schema.yaml`     | JSON Schema for resource profiles               | Same                                     |
| `config/overrides/local.yaml`             | Local dev overrides (lake_root, wandb:disabled) | No loader                                |
| `config/overrides/cluster/ascend.yaml`    | Ascend cluster overrides                        | No loader                                |
| `config/overrides/cluster/cardinal.yaml`  | Cardinal cluster overrides                      | No loader                                |
| `config/overrides/cluster/pitzer.yaml`    | Pitzer cluster overrides                        | No loader                                |
| `config/recipes/final_eval.yaml`          | Final evaluation recipe                         | Only `ablation.yaml` is wired as default |
| `config/recipes/smoke_test.yaml`          | Smoke test recipe                               | Same                                     |
| `config/VALIDATION_CHECKLIST.md`          | 7-item checklist for config validation          | Doc only, no automated checks            |

**Decision needed:** Are these design-ahead scaffolding (intended for future wiring) or
premature additions that should be deleted? The schema files would need explicit
`jsonschema.validate()` call sites. The overrides look like the start of a Hydra-style
system that was never connected. Per project philosophy ("every abstraction must earn its
place"), these should probably be deleted until they have consumers.

---

## 4. Open Design Questions

### 4.1 Analysis runs in-process, not via SLURM

`make_analysis_asset()` in `assets.py` calls `run_analysis_from_spec()` directly in the
dagster worker process — not via a SLURM job. The `analyze-from-spec` CLI command exists
and is registered in `__main__.py`, but the dagster path never invokes it.

This means analysis (embeddings, CKA, loss landscape) runs on whatever machine the dagster
daemon is on, not on a GPU node. If analysis needs GPU (loss landscape definitely does),
this is broken. If analysis is CPU-only, it's fine but should be documented.

**Decision needed:** Should analysis submit a SLURM job like training does? If so, the
`slurm.py` submission logic needs an analysis equivalent of `run_training_job`.

### 4.2 Topology is now Python, not YAML

`topology.py` hardcodes `STAGES`, `_STAGE_DEFS`, `PIPELINE_YAML` as Python dicts. The old
`pipeline.yaml` was deleted. `matrix/axes.yaml` provides `scales`, `model_families`, and
`fusion_methods`, but stage definitions and the DAG are in Python.

This is a hybrid: axes are data-driven (YAML), but topology is code. Adding a new stage
requires editing `topology.py` (Python), not just adding a YAML file.

**Decision needed:** Is this intentional? If the goal was "config as data," the stage
definitions should move back to YAML. If the goal was "topology is code because it has
validation logic," the current approach is fine but should be documented.

### 4.3 Config composition: how do new YAML dirs compose?

The refactor created `models/{family}/base.yaml + scales/{scale}.yaml` and
`fusion/base.yaml + scales/ + methods/`. `TrainingContract.resolve_config_files()` in
`ops.py:77-116` assembles the config chain:

```
stages/{stage}.yaml + models/{family}/base.yaml + models/{family}/scales/{scale}.yaml
```

For fusion:

```
stages/fusion.yaml + fusion/base.yaml + fusion/scales/{scale}.yaml + fusion/methods/{method}.yaml
```

This looks correct but **has not been validated against jsonargparse**. The merge order
matters — later files override earlier ones. If `base.yaml` and `scales/small.yaml` define
the same key, does the scale file win? Needs a parse test.

### 4.4 `defaults/trainer.yaml` replaces root `trainer.yaml`

`cli.py:88` references `defaults/trainer.yaml` as `default_config_files`. This was
previously `trainer.yaml` at the config root. The content and merge semantics should be
verified — is it the same content, or were defaults changed during the move?

### 4.5 CLAUDE.md command auto-discovery claim vs reality

CLAUDE.md says "auto-discovered from `graphids/commands/`" but `__main__.py` uses a
hardcoded `_COMMAND_MODULES` dict. Adding a subcommand requires editing the dict AND
creating the file. The doc is wrong.

---

## 5. Documentation Drift

These docs describe the pre-refactor structure and need updating:

| Doc                                   | Issue                                                                                                                                                                                                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CLAUDE.md`                           | References old files: `pipeline.yaml`, `constants.yaml`, `datasets.yaml`, `resources.yaml`, `trainer.yaml`, old `overlays/`. Says commands are "auto-discovered." Lists per-fusion-method stage YAMLs (`fusion_bandit.yaml`, etc.) that were deleted. |
| `.claude/rules/config-system.md`      | Entire file layout section is stale. References `constants.yaml`, `pipeline.yaml`, `datasets.yaml`, `resources.yaml`, `trainer.yaml`, `write_paths.yaml`, `overlays/`.                                                                                |
| `PLAN.md`                             | Handoff section describes scripts refactor (previous session), not this config/orchestrate/model refactor.                                                                                                                                            |
| `graphids/config/CONFIG_REFERENCE.md` | Added in this refactor (649 lines). Needs verification that it matches the actual current state.                                                                                                                                                      |
| `graphids/config/README.md`           | Added in this refactor (39 lines). Same.                                                                                                                                                                                                              |

---

## 6. Stale Tooling

- **GitNexus index** — 43 stale nodes reported at session start. Model files show at old
  flat locations (`core/models/vgae.py` instead of `core/models/autoencoder/vgae.py`).
  `CheckpointPathIOManager` still indexed in `component.py`. Must re-index before any
  impact analysis is trustworthy: `npx gitnexus analyze`.

---

## 7. Recommended Task Sequence

Priority order based on what blocks what:

### P0 — Blocks all execution

1. **Fix dataset catalog** — either restore `datasets.yaml` or wire up per-file dataset
   configs in `config/datasets/`. `component.py:19` and `datamodule.py` will crash without it.

2. **Fix test imports** — update ~25 model imports and ~10 orchestrate imports to new paths.
   Delete `test_iomanager.py`. Fix mock patch namespaces in dagster tests.

### P1 — Blocks confidence in correctness

3. **Validate config chains parse** — run `python -m graphids.orchestrate validate` on a
   compute node. If it crashes, the `resolve_config_files()` → jsonargparse merge order
   needs debugging.

4. **Decide analysis SLURM path** — if loss landscape needs GPU, `make_analysis_asset` must
   submit via SLURM like training does. Wire `analyze-from-spec` into the dagster path or
   document that analysis is CPU-only.

5. **Run test suite via SLURM** — `scripts/submit.sh tests` to find any remaining import
   breakage not caught by grep.

### P2 — Cleanup

6. **Delete orphaned files** — schema/, overrides/, `allowed_combinations.yaml`,
   `VALIDATION_CHECKLIST.md` unless there's a concrete plan to wire them.

7. **Update docs** — CLAUDE.md, config-system.md, PLAN.md to reflect new structure.

8. **Re-index GitNexus** — `npx gitnexus analyze`.

9. **Delete `graphids/REFACTOR_NOTES_2026-03-31.md`** — its content is now captured here.

10. **Clean stale memories** — per PLAN.md: `project_hydra_config_refactor.md` (Hydra was
    rejected), plus memories that duplicate rules files.
