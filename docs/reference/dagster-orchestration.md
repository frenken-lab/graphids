# Dagster-Native Orchestration Redesign

> Status: **implemented** | Proposed: 2026-03-29 | Audited: 2026-04-02

## Problem (original)

The original `dagster_defs.py` (~513 lines) was a custom orchestration layer written
*inside* dagster. It reimplemented asset factories, config resolution, checkpoint wiring,
and SLURM submission in ad-hoc Python.

## Current Architecture (`graphids/orchestrate/`)

| File | Role |
|------|------|
| `component.py` | Main logic: Component, IOManager, Resource, asset factory, config resolution |
| `__main__.py` | CLI: `run`, `validate` subcommands |
| `planning.py` | Recipe → execution plan (enumerate_assets, StageConfig) |
| `execution.py` | Plan executor |
| `assets.py` | Dagster @asset definitions (make_training_asset, make_analysis_asset) |
| `checks.py` | Dagster freshness/quality checks |
| `analysis.py` | Analysis asset integration |
| `validate.py` | Config chain validation |
| `resolve.py` | ConfigResolver (cross-field validation + audit trail) |
| `resources.py` | ResourceSpec + scale_resources |
| `slurm.py` | sbatch submit, sacct poll |
| `definitions.py` | Dagster entry point — instantiates SlurmTrainingComponent |

### Architecture

```
SlurmTrainingComponent (dg.Component)
├── build_defs() — reads topology + recipe
│   ├── enumerate_assets(topology, recipe) → list[StageConfig]
│   ├── MultiPartitionsDefinition (datasets × seeds)
│   ├── one @asset per StageConfig
│   ├── checkpoint + analysis checks
│   └── returns Definitions with Resource + IOManager
│
├── CheckpointPathIOManager — ckpt path handoff via JSON sidecars
├── SlurmTrainingResource — submit_and_wait → slurm.py
└── ConfigResolver — cross-field validation + audit trail
```

## Decisions

| Decision | Rationale |
|----------|-----------|
| dagster-slurm rejected | Pipes protocol overhead, remote-first design |
| Dagster Pipes rejected | Training is CLI commands, not Pipes-aware Python |
| Custom CheckpointPathIOManager | JSON sidecar for ckpt path handoff |
| dagster Component (`dg.Component`) | YAML-driven config, `dg` CLI discovery |
| .complete marker for skip-if-done | `best_model.ckpt` alone insufficient (stale/killed runs) |

## Cross-references

- History + lessons: `dagster-history.md`
- Open items: `../backlog/open-items.md`
