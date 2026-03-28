# Priority Tier List + Implementation Details

> Status: **active** | Last updated: 2026-03-28

---

## Tier 3: Code consolidation (when you have a clean week)

| #   | What                                       | File(s)                   | Lines   | Details in                                   | Status |
| --- | ------------------------------------------ | ------------------------- | ------- | -------------------------------------------- | ------ |
| 3.1 | Dead code deletion                         | models + preprocessing    | -200    | models-consolidation.md S4, preprocessing S1 | pending |
| 3.2 | `GraphModuleBase` shared base              | `models/_training.py`     | net -90 | models-consolidation.md S2, S3               | pending |
| 3.3 | Delete `configure_optimizers` + wire CLI   | `__main__.py` + 3 modules | net -25 | models-consolidation.md S1                   | pending |
| 3.4 | Preprocessing DataModule conventions       | 3 DataModules             | net +31 | preprocessing-consolidation.md S6            | pending |
| 3.5 | Dissolve `registry.py`                     | `models/`                 | net -73 | models-consolidation.md S6                   | pending |
| 3.6 | Inline `_training.py` single-use utilities | `models/`                 | net -18 | models-consolidation.md S7                   | pending |
| 3.7 | `temporal.py` checkpoint fix               | `models/temporal.py`      | -10     | models-consolidation.md S8                   | pending |

## Tier 4: When writing the paper

| #   | What                                   | File(s)       | Lines        | Details in                                | Status |
| --- | -------------------------------------- | ------------- | ------------ | ----------------------------------------- | ------ |
| 4.1 | ~~Artifacts rewrite~~                  | ~~artifacts/~~| ~~net -220~~ | ~~artifacts-consolidation.md~~            | **DONE** (Analyzer subcommand, 2026-03-28) |
| 4.2 | DQN/Bandit -> LightningModules         | `models/`     | net -120     | models-consolidation.md S5                | pending |
| 4.3 | Memory bloat spike (prefetch thread)   | spike         | experimental | memory_profiling/resource_plan Problem 2  | pending |

## Completed tiers (archived)

| What | Completed | Details |
| ---- | --------- | ------- |
| Config flatten (Hydra -> jsonargparse + flat YAML) | 2026-03-28 | `plans/flatten-model-config.md` |
| Artifacts `analyze` subcommand | 2026-03-28 | `graphids/core/artifacts/analyzer.py` |
| Lightning callback extraction + LightningCLI | 2026-03-27 | `graphids/__main__.py` (GraphIDSCLI) |
| Config system rewrite (Hydra -> jsonargparse) | 2026-03-26 | `graphids/config/` |
| Pipeline deletion (runner, stages, manifest) | 2026-03-26 | `graphids/pipeline/` deleted entirely |
| Codebase cleanup (PyG APIs, Lightning built-ins) | 2026-03-25 | Custom DataLoader/collation replaced |

## Cross-references

| Plan file                                      | Scope                                  | Status |
| ---------------------------------------------- | -------------------------------------- | ------ |
| `plans/models-consolidation.md`                | 13 model files, -287 lines            | proposed (still current) |
| `plans/preprocessing-consolidation.md`         | 8 data files, -179 lines              | proposed (still current) |
| `plans/artifacts-consolidation.md`             | 6 artifact files                       | **superseded** -- Analyzer implemented differently |
| `plans/pipeline-consolidation.md`              | Orchestration + SLURM                  | **superseded** -- pipeline/ deleted, orchestrate/ exists |
| `plans/experiment-sweep-plan.md`               | Ablation + HPO design                  | **partially superseded** -- claims/configs still valid, infra sections stale |
| `plans/flatten-model-config.md`                | Config flatten reference               | completed |
