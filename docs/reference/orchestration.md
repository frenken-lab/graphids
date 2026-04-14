# Orchestration — `graphids/orchestrate/`

> Status: **implemented** | Last refactor: 2026-04-14 (analyze decoupled,
> `.complete` marker retired in favor of checkpoint-authoritative resume)

Pipeline orchestration for the GraphIDS training matrix. `run_pipeline`
executes the 3-stage chain (`autoencoder → supervised → fusion`)
in-process, looping `build → train → evaluate` over each stage within
a single SLURM allocation. Analysis is decoupled — run `python -m
graphids analyze --ckpt-path <p>` after the pipeline. No actor framework,
no cross-node plumbing — the whole pipeline runs in the Python process
that `scripts/slurm/submit.sh pipeline-run` spawns on the compute node.

## Layout

| Module | Role |
|---|---|
| `run.py` | `run_pipeline(config)` — the single driver; no analysis calls |
| `stage.py` | Single-stage primitives: `build(resolved)`, `train(artifacts, resolved)`, `evaluate(artifacts, resolved)`. Shared with the Typer CLI (`cli/training.py`). |
| `resolve.py` | `ResolvedConfig.resolve` classmethod + private `_build_tla_dict`; inline monitor/mode consistency check |
| `_setup.py` | `ensure_spawn`, `touch_marker` |
| `planning.py` | `build_pipeline_stages(cfg)` + `resolve_jsonnet_path(stage)`. Pure planning, no side effects. |
| `config.py` | Frozen data types: `PipelineConfig`, `StageConfig`, `TrainingRunConfig`, `KDEntry`, `ResolvedConfig`, `InstantiatedRun`, `PipelineResult` |
| `analyze.py` | (removed from driver) — `run_single_analysis` now only runs via `python -m graphids analyze` |

## Execution flow

```
pipeline-run  (cli/pipeline.py)
|
+-- PipelineConfig(**kwargs)                            [config.py]
|     -> validates dataset/scale/stages/fusion_method
|
+-- run_pipeline(config) -> PipelineResult              [run.py]
    |
    +-- ensure_spawn()                                  [_setup.py]
    +-- build_pipeline_stages(config) -> list[StageConfig]   [planning.py]
    |
    +-- for each StageConfig (with retry):
    |     +-- ResolvedConfig.resolve(cfg, ...)          [resolve.py]
    |     +-- skip if best_model.ckpt exists                 <-- checkpoint is authoritative
    |     +-- build(resolved)                           [stage.py]
    |     |     -> gc + torch.cuda reset
    |     |     -> instantiate(rendered)                [orchestrate/instantiate.py]
    |     +-- train(artifacts, resolved)                [stage.py]
    |     |     -> wire_file_exporters(run_dir)
    |     |     -> trainer.fit(...)                         <-- resumes from last.ckpt if present
    |     |     -> touch .train_complete
    |     +-- evaluate(artifacts, resolved)             [stage.py]
    |           -> trainer.test(...)
    |           -> touch .test_complete + save predictions
    |
    +-- return PipelineResult(checkpoints, stage_to_asset)
```

Per-stage retries are bounded by `config.max_retries`. A crashed test
leaves no `.test_complete` marker but `best_model.ckpt` may still be on
disk (written mid-training by `ModelCheckpoint`); the ckpt controls
whether the stage re-runs. The old `.complete` marker — a Dagster-era
flag written unconditionally at the end of `evaluate` — was retired when
the pipeline gained direct checkpoint awareness; it was masking OOM /
CUDA crashes by claiming stages were done when they weren't.

## Key decisions

| Decision | Rationale |
|---|---|
| In-process loop, no actor framework | `torchmonarch` was never wired into `pyproject.toml`; the actor scaffold was 600+ lines of dead code. A plain for-loop over stages is all the single-allocation pipeline actually needs. See commit `ad45429` for the full delete. |
| Analyze per-stage, not pipeline-level | A partial chain still leaves usable artifacts behind if a late stage fails. No batched analyzer work to amortize. |
| `build` / `train` / `evaluate` are dumb primitives | No `ResolvedConfig` knowledge, no cache knowledge. Shared verbatim by the `fit`/`test` CLI commands and `run_pipeline`. |
| `ResolvedConfig.resolve` is the exclusive merge path | All override sources (trainer, stage, KD) flow through one classmethod call. |
| Dataset caching sits **below** `build()` | Process-level `get_or_build` in `core/data/cache.py`, keyed on the dataset's `cache_key`. |

## Cross-references

- [`config-architecture.md`](config-architecture.md) — jsonnet + Pydantic layer
- [`write-paths.md`](write-paths.md) — lake layout, `PathContext`, identity hash
- [ADR 0009 — Collapse override handoff chain](../decisions/README.md)
