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
            │       └── Phase 4 (jsonargparse retooling)
            │               └── Phase 5 (Dagster boundaries)
            └── Phase 6 (PyIceberg)  ← independent of 3/4/5, run in parallel
                    └── Phase 7 (sweeps)
```

Phase 6 is fully independent — start it alongside Phase 2 since it only touches artifact writes, not config parsing. Everything else is sequential.

---

## Phase 1 — Jsonnet

**Goal:** Replace YAML chain + `merge_yaml_chain` + override plumbing with
jsonnet. **Full migration, single PR — no shadow path, no dual-write.** Git
history is the rollback. `LightningCLI` stays until Phase 3; jsonargparse is
retooled in Phase 4 for analyzer configs.

---

## Phase 2 — Pydantic Validation Layer

**Goal:** Insert a torch-free Pydantic validation layer between
`render_config(...)` and downstream consumers. Replace
`orchestrate.resolve._convention_errors` (hand-rolled post-hoc linting
over the rendered dict) with real typed `@model_validator` rules so
structural errors, null list fields, monitor-wiring mismatches, and
un-namespaced `class_path` strings die at planning time with an actionable
message.
---

## Phase 3 — Strip LightningCLI

**Goal:** Remove LightningCLI, keep Lightning Trainer.

**NOT verified:**

`WandbLogger` is not constructed in default stage configs; first
production sweep is the first real exercise

---

## Phase 4 — Jsonargparse Retooling

- Upgrade dependency to `jsonargparse[all,shtab,argcomplete]>=4.47.0`
- Switch analyzer configs (`configs/stages/analyze_*.jsonnet`) to Jsonnet
- Update `commands/analyze.py` to use `ArgumentParser(parser_mode="jsonnet")`
  with `--config` so analyzer configs parse as Jsonnet while CLI overrides
  still work (`--analyzer.ckpt_path=...`)
- Use type hints (e.g. `Literal["vgae","gat","fusion"]`) to tighten analyzer
  validation at parse time
- Docs: refresh usage examples and reference tables to point at `.jsonnet`
- ***
  What this means for your stack

| Concern                                                       | jsonnet handles | jsonargparse handles     |
| ------------------------------------------------------------- | --------------- | ------------------------ |
| Config composition, inheritance, mixins                       | ✓               | ✗                        |
| Lazy field computation (`self.lr * 1000`)                     | ✓               | ✗                        |
| Import chaining across files                                  | ✓               | ✗                        |
| Conditional config logic                                      | ✓               | ✗                        |
| ExtVars / TLA injection                                       | ✓               | ✗ (delegates to jsonnet) |
| CLI override on top of rendered config                        | ✗               | ✓                        |
| Typed argument validation (paths, URLs, restricted numbers)   | ✗               | ✓                        |
| Relative path resolution from config location                 | ✗               | ✓                        |
| Argument linking (batch_size → model + datamodule)            | ✗               | ✓                        |
| Class signature introspection (auto-add args from `__init__`) | ✗               | ✓                        |
| Env var override (`APP_LR=1e-4`)                              | ✗               | ✓                        |
| `--print_config` for debugging                                | ✗               | ✓                        |
| Pydantic / dataclass / attrs native support                   | ✗               | ✓                        |

The argument linking feature in particular is probably replacing a chunk of your custom resolver right now — if batch_size or seq_len appears in multiple config sections and you're manually keeping them in sync, link_arguments eliminates that entirely.

## Phase 5 — Dagster Asset Config Boundaries ✓

**Completed 2026-04-05.**

`TrainingAssetConfig(dg.Config)` in `orchestrate/asset_config.py` provides
launch-time overridable knobs (`run_test`, `run_analysis`, `dry_run`).
Asset function returns `dg.Output[str]` with metadata. Checkpoint paths
already flowed via Dagster asset I/O — `upstream_ckpt_paths` in
`TrainingSpec` is populated from asset inputs at resolution time, not from
config. Identity fields stay in `StageConfig` (planner-derived, not
overridable).

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
| jsonargparse                                           | **Keep** (Phase 4)           |
| YAML config chain                                      | **Remove** (Phase 1)         |
| Custom resolver                                        | **Remove** (Phase 1)         |
| DuckDB rebuild script                                  | **Remove** (Phase 6)         |
| `go-jsonnet` binary                                    | **Add** (Phase 1)            |
| stdlib `argparse` entrypoint                           | **Add** (Phase 3, ~20 lines) |
| Pydantic per-asset config models                       | **Add** (Phase 2)            |
| PyIceberg + catalog backend                            | **Add** (Phase 6)            |
