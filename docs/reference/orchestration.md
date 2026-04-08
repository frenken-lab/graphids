# Orchestration — `graphids/orchestrate/`

> Status: **implemented** | Dagster removed: 2026-04-08

Pipeline orchestration for the KD-GAT training matrix. Monarch actors
execute the 3-stage pipeline (autoencoder → supervised → fusion) in a
single SLURM allocation. `ConfigResolver` is the exclusive merge path
that turns a `StageConfig` into a rendered, validated config.

## Layout

| Area | Files | Role |
|---|---|---|
| Root | `contracts.py`, `analysis.py` | `TrainingSpec` + TLA dict, analysis runner |
| `planning/` | `planner.py`, `recipes.py` | Recipe expansion, `StageConfig`, `enumerate_assets` |
| `resolve.py` | (flat module) | `ConfigResolver` + cross-field validation rules |
| `ops/` | `status.py`, `catalog.py`, `finalize.py` | CLI entry points (pipeline-status, catalog rebuild, finalize sidecars) |

## Layered structure (no cycles)

```
LEAVES     planning/ (pure data, Pydantic models)
               │
CONTRACTS  contracts.py ◄── planning
               │
RESOLVE    resolve.py ◄── contracts, planning
               │
ANALYSIS   analysis.py ◄── planning
               │
OPS        ops/ ◄── planning (status), config (catalog, finalize)
```

## Runtime architecture (Monarch path)

```
monarch-run / monarch-sweep (CLI)
│
├── expand_recipe_configs(recipe)            → normalized dict
├── enumerate_assets(TOPOLOGY, recipe)       → list[StageConfig]
│
└── PipelineActor (one per SLURM allocation)
    └── run_stage(stage_cfg, dataset, seed, upstream_ckpts)
        └── ConfigResolver.resolve(cfg, dataset, seed)
            ├── build_tla_dict → typed TLA dict
            ├── apply_resource_overrides → ResourceSpec
            ├── render_config(jsonnet_path, jsonnet_tla)
            ├── validate_config(rendered)  ← Pydantic ValidatedConfig
            ├── validate_stage_config      ← cross-field rules
            └── returns ResolvedConfig
                └── train_entrypoint.run_training(rendered) (in-process)
```

## Key decisions

| Decision | Rationale |
|---|---|
| `ConfigResolver` is the exclusive merge path | All override sources (trainer, stage, KD, resource) flow through one resolver call |
| Monarch over Dagster | Single SLURM allocation for 3-stage pipeline, no inter-job queue wait |
| In-process execution | No JSON envelope serialization boundary — resolver output feeds directly to `instantiate()` |

## Cross-references

- [`config-architecture.md`](config-architecture.md) — jsonnet + Pydantic layer
- [`write-paths.md`](write-paths.md) — lake layout, `PathContext`, identity hash
- [ADR 0009 — Collapse override handoff chain](../decisions/0009-collapse-override-handoffs.md)
