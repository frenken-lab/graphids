# Orchestration — `graphids/orchestrate/`

> Status: **implemented** | Dagster removed: 2026-04-08

Pipeline orchestration for the KD-GAT training matrix. Monarch actors
execute the 3-stage pipeline (autoencoder -> supervised -> fusion) in a
single SLURM allocation. `ResolvedConfig.resolve` is the exclusive merge
path that turns a `StageConfig` into a rendered, validated config.

## Layout

| Module | Role |
|---|---|
| `monarch.py` | `PipelineConfig`, `SweepConfig`, `JobSpec`, `ChainSpec`; `run_chain`, `run_sweep`, `plan_chains`, `build_pipeline_stages`, `chain_job_spec` |
| `actors.py` | `PipelineActor` — `train_stage` + `eval_stage` endpoints; dataset caching across stages |
| `resolve.py` | `ResolvedConfig.resolve` + `_build_tla_dict` + cross-field validation |
| `analysis.py` | Shared analysis runner (called by `eval_stage`) |
| `_setup.py` | `ensure_spawn`, `touch_marker`, `bootstrap_staging` |
| `planning/` | `planner.py`: `StageConfig`, `enumerate_assets`; `recipes.py`: `TrainingRunConfig`, `expand_recipe_configs` |
| `ops/` | `status.py`: `pipeline-status` CLI; `catalog.py`: DuckDB rebuild from OTel traces |

## Layered structure (no cycles)

```
LEAVES     planning/ (pure data, Pydantic models)
               |
RESOLVE    resolve.py <-- planning, config, slurm
               |
ACTOR      actors.py <-- resolve, planning, instantiate
               |
PIPELINE   monarch.py <-- actors, planning, slurm
               |
OPS        ops/ <-- planning (status), config (catalog)
```

## Runtime architecture (Monarch path)

```
monarch-run / monarch-sweep  (cli/_monarch.py)
|
+-- build_pipeline_stages(PipelineConfig)     -> list[StageConfig]   [monarch.py]
|     +-- enumerate_assets(recipe)
|
+-- run_chain(ChainSpec)                                             [monarch.py]
    +-- chain_job_spec(stages) -> JobSpec
    +-- JobSpec.create_job()   -> monarch SlurmJob
    +-- PipelineActor (actors.py) spawned on proc_mesh
        +-- train_stage(stage_config, dataset, seed, upstream_ckpts) -> ckpt_path
        |   +-- ResolvedConfig.resolve(cfg, lake_root, user, dataset, seed, upstream_ckpts)
        |       +-- PathContext(...)
        |       +-- _build_tla_dict(...)          -> typed TLA dict
        |       +-- apply_resource_overrides(...)  -> ResourceSpec
        |       +-- render(jsonnet_path, tla)       -> dict
        |       +-- validate_config(rendered)       -> ValidatedConfig
        |       +-- _validate_cross_fields(...)
        |   -> instantiate(resolved.rendered) -> run
        |   -> run.trainer.fit(...)
        |
        +-- eval_stage(stage_config, ...) -- test + analyze + phase markers (lenient)
```

`run_sweep` fans out `plan_chains` results over a `ThreadPoolExecutor`,
one `run_chain` call per `ChainSpec`.

## Key decisions

| Decision | Rationale |
|---|---|
| `ResolvedConfig.resolve` is the exclusive merge path | All override sources (trainer, stage, KD, resource) flow through one call |
| Monarch over Dagster | Single SLURM allocation for 3-stage pipeline; no inter-job queue wait |
| In-process execution | Resolver output feeds directly to `instantiate()` — no serialization boundary |
| Dataset caching in actor | `PipelineActor` holds CPU copies across stages, avoiding redundant preprocessing I/O |

## Cross-references

- [`config-architecture.md`](config-architecture.md) — jsonnet + Pydantic layer
- [`write-paths.md`](write-paths.md) — lake layout, `PathContext`, identity hash
- [ADR 0009 — Collapse override handoff chain](../decisions/README.md)
