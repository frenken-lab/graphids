# Orchestration — `graphids/orchestrate/`

> Status: **implemented** | Layout audited: 2026-04-04

Dagster-native pipeline orchestrator for the KD-GAT training matrix. A
`SlurmTrainingComponent` reads a recipe YAML, expands it against the stage
topology, and produces one partitioned asset per unique
`(stage, model_type, scale, fusion_method, kd_variant)` combination. Each
asset submits a single SLURM job that runs train → test → analyze.

## Files (10, 1,171 LOC)

| File | LOC | Role |
|---|---:|---|
| `__init__.py` | 11 | Package docstring |
| `__main__.py` | 29 | CLI stub (validation moved into resolver gates) |
| `definitions.py` | 35 | Dagster discovery entry (via `pyproject.toml`); configures logging, instantiates `SlurmTrainingComponent`, calls `build_defs_for_component` |
| `component.py` | 138 | `SlurmTrainingComponent` (`dg.Component`) + `SlurmTrainingResource` (`dg.ConfigurableResource`). Assembles `Definitions(assets, asset_checks, resources, executor)` |
| `planning.py` | 187 | Pure data: `enumerate_assets(pipeline, recipe) → list[StageConfig]` (StageConfig lives in `graphids.config.shared`). Two-pass expansion with canonical dedup, identity hashing, KD teacher resolution. No dagster imports |
| `resolve.py` | 312 | `ConfigResolver` — the exclusive merge path (ADR 0009). Takes a `StageConfig` and produces `ResolvedConfig(spec, resources, paths, audit)`. Runs override merge + Pydantic gates (`validate_config` + stage cross-field validation) in one pass |
| `assets.py` | 195 | `make_training_asset(cfg)` factory + shared dagster-context helpers (`partition_keys`, `paths_for_context`, `_runtime_lake_root`, `_runtime_user`, `_touch_complete`). One `@dg.asset` per `StageConfig`; bundles train→test→analyze into a single SLURM job |
| `checks.py` | 124 | `make_asset_checks(cfg_lookup)` — one `@dg.multi_asset_check` op per training asset, emitting a blocking `checkpoint_complete_*` check and a non-blocking `analysis_complete_*` check (where supported) with `can_subset=True` |
| `analysis.py` | 48 | Analysis spec/output helpers: `supports_analysis`, `build_analysis_spec`, `output_status`, `ANALYSIS_MANIFEST_NAME`. Shared by `assets.py` and `checks.py` |

## Layered structure (no cycles)

```
LEAVES     planning.py (pure data)
               │
CONTRACTS  analysis.py ◄── planning         resolve.py ◄── planning
               │                                │
FACTORIES  assets.py (uses analysis, planning, resolve)
               │
           checks.py (uses analysis, assets, planning)
               │
HUB        component.py (uses assets, checks, planning)
               │
ENTRIES    definitions.py
               │
           (dagster dg CLI)
```

## Runtime architecture

```
SlurmTrainingComponent (dg.Component)
│
├── build_defs(context)
│   ├── read KD_GAT_RECIPE env var → recipe YAML
│   ├── expand_recipe_configs(recipe)            → normalized dict
│   ├── enumerate_assets(PIPELINE_YAML, recipe)  → list[StageConfig]
│   ├── MultiPartitionsDefinition(dataset × seed)
│   ├── assets.make_training_asset(cfg)          → @dg.asset (one per StageConfig)
│   ├── checks.make_asset_checks(cfg_lookup)     → one multi_asset_check per asset
│   └── Definitions(assets, asset_checks, resources, executor=multiprocess)
│
├── SlurmTrainingResource (dg.ConfigurableResource)
│   └── submit_and_wait → SubprocessSlurmJobClient.run_training_job
│       ├── writes TrainingSpec JSON envelope to shared filesystem
│       ├── sbatch → SLURM queue
│       └── polls sacct until terminal state
│
└── Per-materialization (assets._train body):
    └── ConfigResolver.resolve_and_validate(cfg, dataset, seed)
        ├── merges trainer + stage + KD + resource overrides
        ├── render_config(jsonnet_path, jsonnet_tla) → dict
        ├── validate_config(rendered) → ValidatedConfig (Pydantic gate)
        ├── validate_stage_config(...) (workers ≤ cpus-1, curriculum epoch sync, GPU partition, RL dead config)
        └── returns ResolvedConfig → submit_and_wait
```

## Config flow — the 3-stage chain (ADR 0009)

For the complete end-to-end validation flow, override sources, and the
planning-time guarantees that replace the old 9-stage validation desert,
see [`3-chain.md`](3-chain.md).

## Key decisions

| Decision | Rationale |
|---|---|
| `dg.Component` over raw `Definitions` | YAML-driven config, `dg` CLI discovery, component reload semantics |
| `fs_io_manager` over custom IOManager | Training assets return checkpoint paths as `str` outputs; built-in `fs_io_manager` handles JSON serialization fine. No need for a custom sidecar manager |
| `ConfigResolver` is the exclusive merge path | Replaces the old 9-stage override handoff. All override sources (trainer, stage, KD, resource) flow through one resolver call, catching typos pre-SLURM |
| `@dg.multi_asset_check` over paired `@dg.asset_check` | One op per asset emits both checks, sharing `paths_for_context` setup. `can_subset=True` lets dagster target individual checks |
| `.complete` marker for skip-if-done | `best_model.ckpt` alone is insufficient — stale/killed runs can leave partial checkpoints. `_touch_complete` uses NFS-safe fsync(file) + fsync(dir) |
| `train→test→analyze` in one SLURM job | Avoids dagster's per-step scheduling overhead on SLURM (queue wait + GPU startup is the dominant cost) |
| `dagster-slurm` rejected | Remote-first design, Pipes protocol overhead — we don't need Pipes for CLI training scripts |
| `Dagster Pipes` rejected | Training entry points are CLI commands, not Pipes-aware Python |

## Cross-references

- [`3-chain.md`](3-chain.md) — dagster inputs, override flow, validation gates
- [`orchestration-risks.md`](orchestration-risks.md) — known fragile / complex areas, recommended fixes
- [`config-architecture.md`](config-architecture.md) — jsonargparse + YAML layer
- [`write-paths.md`](write-paths.md) — lake layout, `PathContext`, identity hash
- [ADR 0006 — Dagster integration](../decisions/0006-dagster-integration.md)
- [ADR 0009 — Collapse override handoff chain](../decisions/0009-collapse-override-handoffs.md)
