# Orchestration — `graphids/orchestrate/`

> Status: **implemented** | Refactor: 2026-04-10

Pipeline orchestration for the GraphIDS training matrix. `run_pipeline`
executes the 3-stage chain (`autoencoder → supervised → fusion`)
in-process, looping `build → train → evaluate → analyze` over each
stage within a single SLURM allocation. No actor framework, no cross-
node plumbing — the whole pipeline runs in the Python process that
`scripts/slurm/submit.sh pipeline-run` spawns on the compute node.

## Layout

| Module | Role |
|---|---|
| `run.py` | `PipelineConfig`, `build_pipeline_stages`, `run_pipeline(config)` — the single driver that sees every layer |
| `stage.py` | Single-stage primitives: `build(rendered, validated)`, `train(artifacts, *, run_dir, ckpt_file, stage, resume_from)`, `evaluate(artifacts, *, run_dir, ckpt, stage)`. No `ResolvedConfig` dependency — shared with the Typer CLI (`cli/_training.py`). |
| `analyze.py` | `run_single_analysis(spec)` — writes analysis artifacts + manifest sidecar for one checkpoint. Called per-stage from `run_pipeline`. |
| `resolve.py` | `ResolvedConfig.resolve` classmethod + private `_build_tla_dict`; inline monitor/mode consistency check |
| `_setup.py` | `ensure_spawn`, `touch_marker` |
| `planning.py` | `build_pipeline_stages(cfg)` + `resolve_jsonnet_path(stage)`. Pure planning, no side effects. |
| `config.py` | Frozen data types: `PipelineConfig`, `StageConfig`, `TrainingRunConfig`, `KDEntry`, `ResolvedConfig`, `InstantiatedRun`, `PipelineResult` |

## Execution flow

```
pipeline-run  (cli/_pipeline.py)
|
+-- PipelineConfig(**kwargs)                            [run.py]
|     -> validates dataset/scale/stages/fusion_method
|
+-- run_pipeline(config) -> PipelineResult              [run.py]
    |
    +-- ensure_spawn()                                  [_setup.py]
    +-- build_pipeline_stages(config) -> list[StageConfig]   [planning.py]
    |
    +-- for each StageConfig (with retry):
    |     +-- ResolvedConfig.resolve(cfg, ...)          [resolve.py]
    |     +-- skip if complete marker present
    |     +-- build(rendered, validated)                [stage.py]
    |     |     -> gc + torch.cuda reset
    |     |     -> instantiate(rendered)                [graphids/instantiate.py]
    |     +-- train(artifacts, ...)                     [stage.py]
    |     |     -> wire_file_exporters(run_dir)
    |     |     -> trainer.fit(...)
    |     |     -> touch .train_complete
    |     +-- evaluate(artifacts, ...)                  [stage.py]
    |     |     -> trainer.test(...)
    |     |     -> touch .test_complete + .complete (lenient)
    |     +-- if analyzable (vgae/dgi/gat):
    |         run_single_analysis(spec)                 [analyze.py]
    |         -> touch .analyze_complete (lenient)
    |
    +-- return PipelineResult(checkpoints, analyzed_assets, stage_to_asset)
```

Per-stage retries are bounded by `config.max_retries`; eval and
analyze failures are logged as warnings and don't fail the chain. The
complete-marker skip-check means a resumed run picks up where the
last one stopped without rebuilding state.

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
