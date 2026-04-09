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
       +- _build_tla_dict(cfg, ...)        -> typed TLA dict
       +- get_resources / apply_resource_overrides
       +- render(jsonnet_path, tla)
       +- validate_config(rendered)        -> ValidatedConfig  (Pydantic)
       +- _validate_cross_fields(...)      -> num_workers<=cpus-1, epoch sync
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
| `num_workers > cpus_per_task - 1` | `_validate_cross_fields` | `orchestrate/resolve.py` |
| `CurriculumDataModule.max_epochs != trainer.max_epochs` | `_validate_cross_fields` | `orchestrate/resolve.py` |
| Stage monitor family mismatch (val_acc vs val_loss) | `ResolvedConfig.resolve` warning | `orchestrate/resolve.py` |

**Structural failures are caught before the SLURM job starts.** The Monarch
actor runs `ResolvedConfig.resolve()` in-process and the result flows
directly into `instantiate()`.

## Key files

| File | Role |
|---|---|
| `graphids/orchestrate/planning/recipes.py` | `TrainingRunConfig`, `KDEntry`, `expand_recipe_configs` |
| `graphids/orchestrate/planning/planner.py` | `StageConfig`, `enumerate_assets` |
| `graphids/orchestrate/resolve.py` | `ResolvedConfig.resolve`, `_build_tla_dict`, `_validate_cross_fields` |
| `graphids/orchestrate/actors.py` | `PipelineActor` — `train_stage` / `eval_stage` endpoints |
| `graphids/orchestrate/monarch.py` | `PipelineConfig`, `SweepConfig`, `plan_chains`, `run_chain`, `run_sweep` |
| `graphids/config/schemas.py` | `ValidatedConfig`, `validate_config` |
| `graphids/config/topology.py` | Stage DAG, identity keys, import-time assertions |
| `configs/stages/autoencoder.jsonnet` | Stage 1 jsonnet function |
| `configs/stages/supervised.jsonnet` | Stage 2 jsonnet function |
| `configs/stages/fusion.jsonnet` | Stage 3 jsonnet function |
