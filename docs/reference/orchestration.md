# Orchestration — `graphids/orchestrate/`

> Status: **implemented** | Refactor: 2026-04-10 (session 42)

Pipeline orchestration for the KD-GAT training matrix. A single
Monarch `SlurmJob` hosts a `PipelineActor` that executes the 3-stage
pipeline (`autoencoder → supervised → fusion`) one stage at a time.
Each file owns one verb at one level; composition happens only in the
explicit `run_*` drivers.

## Layout

| Module | Role |
|---|---|
| `run.py` | `PipelineConfig` schema, `build_pipeline_stages`, `PipelineResult`, and the top-level `run_pipeline(config, job_spec)` driver — the only module that sees every layer |
| `allocate.py` | `JobSpec`, `build_slurm_job`, `spawn_actor`, `configure_monarch` — SLURM allocation with zero pipeline knowledge |
| `chain.py` | `run_chain(actor, stages, …) → ChainResult` — pure loop over `train_stage` then `eval_stage`, decoupled from the SlurmJob lifecycle |
| `analyze.py` | Pipeline-level `analyze(actor, stages, chain, …)` + `run_single_analysis(spec)` — runs once after `run_chain` returns, over the full dict of checkpoints |
| `stage.py` | Single-stage primitives: `build(resolved)`, `train(artifacts, resolved)`, `evaluate(artifacts, resolved, ckpt)`, `run_stage(resolved)` driver |
| `actors.py` | `PipelineActor` — thin Monarch endpoint wrapper around `stage.py` primitives (`train_stage`, `eval_stage`, `analyze_stage` endpoints) |
| `resolve.py` | `ResolvedConfig.resolve` classmethod + private `_build_tla_dict`; inline monitor/mode consistency check |
| `_setup.py` | `ensure_spawn`, `touch_marker` |
| `planning/` | `planner.py`: `StageConfig`, `enumerate_assets`, `resolve_jsonnet_path`; `recipes.py`: `TrainingRunConfig`, `KDEntry`, `expand_recipe_configs`, `check_in` |

## Layered structure (no cycles)

```
LEAVES     planning/  (pure data, Pydantic models)
               |
RESOLVE    resolve.py          <-- planning, config, slurm
               |
STAGE      stage.py            <-- resolve, instantiate  (build, train, evaluate, run_stage)
               |
ACTOR      actors.py           <-- stage  (Monarch endpoint wrapper)
               |
CHAIN      chain.py            <-- planning  (run_chain over actor endpoints)
               |
ANALYZE    analyze.py          <-- chain, planning  (pipeline-level driver)
               |
ALLOCATE   allocate.py         <-- actors, slurm
               |
DRIVER     run.py              <-- allocate, chain, analyze, planning
```

Each layer composes only its Layer N+1 peers. `run_pipeline` is the
only module that sees the full picture.

## Runtime architecture

```
monarch-run  (cli/_monarch.py)
|
+-- PipelineConfig(**kwargs)                                         [run.py]
|     -> validates dataset/scale/stages/fusion_method
|
+-- JobSpec(partition=..., time=..., mem=..., cpus=...)              [allocate.py]
|
+-- run_pipeline(config, job_spec) -> PipelineResult                 [run.py]
    |
    +-- build_pipeline_stages(config) -> list[StageConfig]           [run.py]
    |     +-- enumerate_assets(recipe)                               [planning/planner.py]
    |
    +-- configure_monarch()                                          [allocate.py]
    +-- build_slurm_job(job_spec) -> SlurmJob                        [allocate.py]
    |     +-- patch_clusterscope_for_osc()                           [graphids/_slurm.py]
    +-- spawn_actor(job, gpus_per_node, lake_root) -> PipelineActor  [allocate.py]
    |
    +-- run_chain(actor, stages, dataset, seed, max_retries)         [chain.py]
    |   | -> ChainResult(checkpoints, stage_to_asset)
    |   |
    |   +-- for each stage: actor.train_stage.call_one(...)          [actors.py + stage.py]
    |   |     +-- ResolvedConfig.resolve(cfg, ...)                   [resolve.py]
    |   |     +-- build(resolved) -> InstantiatedRun                 [stage.py]
    |   |     |     -> gc.collect + torch.cuda.empty_cache
    |   |     |     -> instantiate(resolved.rendered)                [graphids/instantiate.py]
    |   |     +-- train(artifacts, resolved) -> ckpt_path            [stage.py]
    |   |           -> wire_file_exporters(run_dir)
    |   |           -> trainer.fit(model, datamodule)
    |   |           -> touch_marker(run_dir/.train_done)
    |   |
    |   +-- for each stage: actor.eval_stage.call_one(...)           [actors.py + stage.py]
    |       +-- build(resolved)
    |       +-- evaluate(artifacts, resolved, ckpt) -> metrics        [stage.py]
    |       |     -> trainer.test(...)
    |       |     -> touch_marker(run_dir/.test_done)
    |       +-- touch_marker(run_dir/.complete)
    |
    +-- analyze(actor, stages, chain, dataset, seed)                 [analyze.py]
    |     +-- for analyzable stages (vgae/dgi/gat):
    |         actor.analyze_stage.call_one(...)                      [actors.py]
    |           +-- run_single_analysis(spec)                        [analyze.py]
    |                 -> Analyzer(...).run()
    |                 -> write analysis_manifest.json
    |                 -> touch_marker(run_dir/.analyze_done)
    |
    +-- finally: job.kill()                                          [run.py]
```

Train-stage retries are bounded by `config.max_retries`; eval-stage
and analyze-stage failures are logged as warnings and don't fail the
chain.

## Key decisions

| Decision | Rationale |
|---|---|
| Each file owns one verb at one level | No function composes two of its peers; composition happens only in explicit drivers. See `plans/kd-gat-orchestrate-refactor.md`. |
| Dataset caching sits **below** `build()` | Process-level `get_or_build` in `core/data/cache.py`, keyed on the dataset's `cache_key`. No actor-side dataset state. |
| `analyze` is pipeline-level, not per-stage | Runs once after `run_chain` returns over the full dict of trained checkpoints; fixed-cost setup amortizes across stages. |
| `build` is dumb | Pure importlib class_path instantiation (plus GPU reset) — no cache knowledge, no dataset knowledge. |
| `ResolvedConfig.resolve` is the exclusive merge path | All override sources (trainer, stage, KD) flow through one classmethod call. |
| Monarch over Dagster | Single SLURM allocation for 3-stage pipeline; no inter-job queue wait. |

## Cross-references

- [`config-architecture.md`](config-architecture.md) — jsonnet + Pydantic layer
- [`write-paths.md`](write-paths.md) — lake layout, `PathContext`, identity hash
- [ADR 0009 — Collapse override handoff chain](../decisions/README.md)
- `plans/kd-gat-orchestrate-refactor.md` — design decisions for this layout
