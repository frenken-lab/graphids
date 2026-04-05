# Config & Artifact Stack Migration Plan

## Context

**Current pain:** 3-chain YAML + custom resolver + LightningCLI + jsonargparse + file-based artifact catalog rebuilt by DuckDB query.

**Target:** jsonnet (composition) → argparse entrypoint → Pydantic validation → Lightning Trainer → PyIceberg artifact catalog → DuckDB queries.

---

## Dependency Graph

```
Phase 1 (jsonnet)
    └── Phase 2 (Pydantic)
            ├── Phase 3 (strip LightningCLI)
            │       └── Phase 4 (strip jsonargparse)
            │               └── Phase 5 (Dagster boundaries)
            └── Phase 6 (PyIceberg)  ← independent of 3/4/5, run in parallel
                    └── Phase 7 (sweeps)
```

Phase 6 is fully independent — start it alongside Phase 2 since it only touches artifact writes, not config parsing. Everything else is sequential.

---

## Phase 1 — Jsonnet

**Goal:** Replace YAML chain + `merge_yaml_chain` + override plumbing with
jsonnet. **Full migration, single PR — no shadow path, no dual-write.** Git
history is the rollback. `jsonargparse` / `LightningCLI` stay (Phases 3–4
strip those).

**Detailed plan:** [`docs/phase1_jsonnet.md`](./phase1_jsonnet.md) — commit-by-commit
migration order, shim design, one-shot parity gate, gotchas, verification.

High-level shape:

1. Install `go-jsonnet` binary (not Python wrapper — faster, no JVM)
2. Port the 20 YAML files that feed `merge_yaml_chain` to jsonnet under `configs/` (repo root)
3. Lazy fields (dataset, seed, run_dir, upstream ckpts, KD auxiliaries, recipe overrides) become TLAs; recipe expansion + planning stay in Python
4. Add `graphids/config/jsonnet.py::render_config(jsonnet_path, tla_dict) → dict`
5. **One-shot parity gate** in Commit 2: `test_jsonnet_parity.py` diffs `render_config(...)` against `merge_yaml_chain(...)` across every recipe chain. Deleted in the same commit that deletes `merge_yaml_chain`.
6. Swap `TrainingSpec`, `TrainingContract`, `StageConfig`, `ConfigResolver`, `train_entrypoint`, `_lightning._BOOTSTRAP`, `cli.run_lightning` (adds `.jsonnet` preprocessor for dev path)
7. Delete `graphids/config/{stages,models,fusion,defaults/trainer.yaml}/`, `merge_yaml_chain`, `deep_merge`, `apply_dotted_overrides`, `to_override_dict`, `resolve_config_files`, `default_config_files`

**Exit criteria:**

- `grep -r merge_yaml_chain graphids/ tests/` → empty
- `ls graphids/config/stages/` → no such directory
- `python -m graphids.orchestrate validate` passes on ablation/smoke_test/final_eval
- `dg launch smoke_test` runs one asset per stage (autoencoder, normal, curriculum, fusion) to COMPLETED
- `python -m graphids fit --config configs/stages/autoencoder.jsonnet --trainer.max_epochs 1` runs on gpudebug
- Net LOC change is **negative** (jsonnet absorbs Python plumbing)

---

## Phase 2 — Pydantic Validation Layer

---

## Phase 3 — Strip LightningCLI

**Goal:** Remove LightningCLI, keep Lightning Trainer.

1. Write explicit `train.py` entrypoint — stdlib `argparse` for `--config` and `--tla`, calls `render_config`, calls `model_validate`, instantiates model/datamodule/trainer directly:
   ```python
   parser = argparse.ArgumentParser()
   parser.add_argument("--config", required=True)
   parser.add_argument("--tla", nargs="*", default=[])
   ```
2. Make `LightningModule.__init__` take flat explicit typed args (not `**kwargs` or namespace objects)
3. Verify `save_hyperparameters()` still works — call it explicitly with a flat dict, not relying on CLI injection
4. Fix `load_from_checkpoint` — pass constructor args explicitly at load time:
   ```python
   model = MyModel.load_from_checkpoint(ckpt_path, lr=cfg.lr, strict=True)
   ```
5. Delete `LightningCLI` instantiation
6. Run one full pretrain + finetune cycle end-to-end through new entrypoint

**Exit criteria:** SLURM jobs launch via `python train.py --config configs/pretrain.jsonnet --tla partition=gpu` with no LightningCLI in the call stack.

---

## Phase 4 — Strip jsonargparse

**Goal:** Remove the last jsonargparse dependency.

1. Audit for any remaining `jsonargparse` imports outside LightningCLI — list them
2. Replace any standalone `ArgumentParser` from jsonargparse with stdlib `argparse`
3. Replace any `add_class_arguments` / `instantiate_classes` usage with explicit Pydantic `model_validate` + direct `__init__` calls
4. `pip uninstall jsonargparse` — fix any import errors
5. Run full test suite

**Exit criteria:** `grep -r jsonargparse .` returns nothing.

---

## Phase 5 — Dagster Asset Config Boundaries

**Goal:** Each Dagster asset owns its config slice; checkpoint paths flow as asset outputs, not config fields.

1. Define per-asset `Config` classes subclassing Dagster's `Config` (Pydantic-compatible) — launch-time-known fields only
2. Checkpoint path returned as asset output (`-> str`), received as upstream input by downstream asset — remove from all config models:

   ```python
   @asset
   def pretrain(config: PretrainConfig) -> str:
       ...
       return checkpoint_path   # artifact, not config

   @asset
   def finetune(config: FinetuneConfig, pretrain: str) -> str:
       # pretrain is the checkpoint — arrives via Dagster, not config
       ...
   ```

3. SLURM job receives `config_path` + `tla` overrides as env vars set by Dagster, not baked into the DAG definition
4. Add sensors for long-running jobs if non-blocking asset execution is needed

**Exit criteria:** `FinetuneConfig` has no `checkpoint` field. Checkpoint arrives via Dagster asset dependency.

---

## Phase 6 — PyIceberg Catalog

**Goal:** Replace file dump + DuckDB rebuild script with a structured catalog written at job completion.

> This phase is independent — run it in parallel with Phases 2–5.

1. Stand up PyIceberg catalog backend:
   - SQLite locally for dev
   - Postgres for production (reuse existing Dagster Postgres if available)

2. Define Iceberg schemas for artifact types:

   | Table         | Key Fields                                                              |
   | ------------- | ----------------------------------------------------------------------- |
   | `experiments` | `run_id`, `config_hash`, `jsonnet_path`, `dagster_run_id`, `created_at` |
   | `checkpoints` | `run_id`, `epoch`, `val_loss`, `artifact_path`, `parent_run_id`         |
   | `metrics`     | `run_id`, `step`, `metric_name`, `value`                                |
   | `lineage`     | `run_id`, `parent_run_id`, `config_hash`, `artifact_path`               |

3. Write `write_run_metadata(run_id, cfg, results)` — called by Dagster asset after SLURM job completes, the **only write site**:

   ```python
   table.append(pa.Table.from_pydict({
       "run_id": [run_id],
       "config_hash": [hash(str(cfg))],
       "artifact_path": [checkpoint_path],
       ...
   }))
   ```

4. Port existing DuckDB rebuild queries to read from Iceberg via `.to_duckdb()` — verify output matches:

   ```python
   conn = catalog.load_table("ml.experiments").scan(
       row_filter="val_loss < 0.1"
   ).to_duckdb("experiments")
   conn.execute("SELECT run_id, artifact_path FROM experiments ORDER BY val_loss")
   ```

5. Delete the rebuild script

**Exit criteria:** "What config produced this checkpoint?" and "What checkpoints came from sweep X?" are answerable via SQL with no file-crawling.

---

## Phase 7 — Sweep Integration

---

## What Is Kept, Removed, and Added

|                                                        | Action                       |
| ------------------------------------------------------ | ---------------------------- |
| Lightning `Trainer`, DDP, callbacks, `LightningModule` | **Keep**                     |
| DuckDB for querying                                    | **Keep**                     |
| LightningCLI                                           | **Remove** (Phase 3)         |
| jsonargparse                                           | **Remove** (Phase 4)         |
| YAML config chain                                      | **Remove** (Phase 1)         |
| Custom resolver                                        | **Remove** (Phase 1)         |
| DuckDB rebuild script                                  | **Remove** (Phase 6)         |
| `go-jsonnet` binary                                    | **Add** (Phase 1)            |
| stdlib `argparse` entrypoint                           | **Add** (Phase 3, ~20 lines) |
| Pydantic per-asset config models                       | **Add** (Phase 2)            |
| PyIceberg + catalog backend                            | **Add** (Phase 6)            |
