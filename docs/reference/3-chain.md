# 3-Stage Training Chain

The three stages are: **autoencoder -> supervised -> fusion**.

## Data flow (Monarch path)

```
CLI (monarch-run / monarch-sweep)
|
+- expand_recipe_configs(raw_recipe)       -> normalized dict
+- enumerate_assets(recipe)                -> list[StageConfig]
|     StageConfig: stage, model_type, scale, identity,
|     trainer_overrides, stage_overrides, kd_overrides,
|     resource_overrides, upstream_asset_names
|     -- graphids/orchestrate/planning/planner.py
|
+- PipelineActor.train_stage(stage_config, dataset, seed, upstream_ckpts)
   +- ResolvedConfig.resolve(cfg, ...)     -- orchestrate/resolve.py
       +- PathContext(...)                 -- config/topology.py
       +- _build_tla_dict(cfg, ...)        -> typed TLA dict (private)
       +- render(jsonnet_path, tla)        -- config/jsonnet.py
       +- validate_config(rendered)        -> ValidatedConfig  (Pydantic)
       +- monitor/mode consistency check   -> inline log warning
       +- returns ResolvedConfig(paths, validated, rendered)
           +- instantiate(rendered, validated=...) -> fit
```

No JSON envelope, no serialization boundary. Resolver output feeds
directly to `graphids.instantiate.instantiate`.

## Where validation catches what

| Failure mode | Caught at | Where |
|---|---|---|
| Invalid stage, scale, fusion_method, conv_type, loss_fn | `TrainingRunConfig` construction | `planning/recipes.py:74-85` |
| KD alpha out of [0,1], invalid teacher_scale | `KDEntry` validators | `planning/recipes.py:54-67` |
| Missing model config file | Import time | `config/topology.py` assertions |
| Missing identity keys for a stage | `compute_identity_hash` | `config/topology.py` |
| `pool_aggrs`, `hidden_dims`, `auxiliaries` as null | `ValidatedConfig._no_null_list_fields` | `config/schemas.py` |
| `LearningRateMonitor` with `trainer.logger=false` | `ValidatedConfig._lr_monitor_requires_logger` | `config/schemas.py` |
| `checkpoint` + `early_stopping` monitor/mode mismatch | `ValidatedConfig` via `CallbacksSection` | `config/schemas.py` |
| `data.class_path` / `model.class_path` not namespaced | `ValidatedConfig._class_paths_namespaced` | `config/schemas.py` |
| Extra top-level key in rendered dict | `ValidatedConfig(extra="forbid")` | `config/schemas.py` |
| Stage monitor family mismatch (val_acc vs val_loss) | `ResolvedConfig.resolve` inline log warning | `orchestrate/resolve.py` |

**Structural failures are caught before the SLURM job starts.** The Monarch
actor runs `ResolvedConfig.resolve()` in-process and the result flows
directly into `instantiate()`.

## Key files

| File | Role |
|---|---|
| `graphids/orchestrate/planning/recipes.py` | `TrainingRunConfig`, `KDEntry`, `expand_recipe_configs` |
| `graphids/orchestrate/planning/planner.py` | `StageConfig`, `enumerate_assets` |
| `graphids/orchestrate/resolve.py` | `ResolvedConfig.resolve` (classmethod), private `_build_tla_dict` |
| `graphids/orchestrate/stage.py` | `build`, `train`, `evaluate`, `run_stage` (single-stage primitives) |
| `graphids/orchestrate/actors.py` | `PipelineActor` — thin endpoint wrapper (`train_stage` / `eval_stage` / `analyze_stage`) |
| `graphids/orchestrate/chain.py` | `run_chain(actor, stages, …) → ChainResult` |
| `graphids/orchestrate/allocate.py` | `JobSpec`, `build_slurm_job`, `spawn_actor` |
| `graphids/orchestrate/analyze.py` | pipeline-level `analyze` + `run_single_analysis` |
| `graphids/orchestrate/run.py` | `PipelineConfig`, `build_pipeline_stages`, `run_pipeline` driver |
| `graphids/config/schemas.py` | `ValidatedConfig`, `validate_config` |
| `graphids/config/topology.py` | Stage DAG, identity keys, import-time assertions |
| `configs/stages/autoencoder.jsonnet` | Stage 1 jsonnet function |
| `configs/stages/supervised.jsonnet` | Stage 2 jsonnet function |
| `configs/stages/fusion.jsonnet` | Stage 3 jsonnet function |
