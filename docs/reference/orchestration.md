# Orchestration — `graphids/orchestrate/`

> Status: **implemented** | Layout audited: 2026-04-04

Dagster-native pipeline orchestrator for the KD-GAT training matrix. A
`SlurmTrainingComponent` reads a recipe YAML, expands it against the stage
topology, and produces one partitioned asset per unique
`(stage, model_type, scale, fusion_method, kd_variant)` combination. Each
asset submits a single SLURM job that runs train → test → analyze.

## Layout

| Area | Files | Role |
|---|---|---|
| Root | `definitions.py`, `analysis.py`, `__main__.py` | Dagster discovery entry + analysis helpers + CLI stub |
| `dagster/` | `component.py`, `resources.py`, `assets.py`, `checks.py`, `asset_config.py`, `runtime.py` | Dagster-facing component, resources, asset/check factories, runtime helpers |
| `planning/` | `planner.py`, `recipes.py`, `shared.py` | Pure planning + recipe expansion + `StageConfig` (no Dagster imports) |
| `resolve/` | `resolver.py`, `cross_field.py` | Config resolution + cross-field validation |
| `contracts/` | `__init__.py` | `TrainingSpec` + envelope helpers |
| `ops/` | `entrypoint.py`, `status.py`, `catalog.py`, `finalize.py` | CLI entry points (from-spec, pipeline-status, catalog rebuild, finalize sidecars) |

## Layered structure (no cycles)

```
LEAVES     planning/ (pure data)
               │
CONTRACTS  analysis.py ◄── planning         resolve/ ◄── planning
               │                                │
FACTORIES  dagster/assets.py (uses analysis, planning, resolve)
               │
           dagster/checks.py (uses analysis, assets, planning)
               │
HUB        dagster/component.py (uses assets, checks, planning)
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
│   ├── dagster.assets.make_training_asset(cfg)  → @dg.asset (one per StageConfig, slurm injected via ResourceParam)
│   ├── dagster.checks.make_asset_checks(cfg_lookup) → one multi_asset_check per asset
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
