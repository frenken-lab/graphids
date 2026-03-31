# Priority Tier List + Implementation Details

> Status: **active** | Last updated: 2026-03-30

## Tier 3: Code consolidation (when you have a clean week)

| # | What | File(s) | Est. delta | Plan reference | Status |
|---|------|---------|------------|----------------|--------|
| 3.1 | Dead code deletion (`fuse()` methods) | `fusion_baselines.py` | -14 | models-consolidation.md §4 | pending |
| 3.2 | `GraphModuleBase` shared base | `models/_training.py` | net -50 | models-consolidation.md §2, §3 | pending |
| 3.3 | Delete `configure_optimizers` + wire CLI | `cli.py` + GAT/DGI/VGAE | net -15 | models-consolidation.md §1 | pending |
| 3.4 | Preprocessing DataModule conventions | 3 DataModules | net +31 | preprocessing-consolidation.md §6 | pending |
| 3.5 | Dissolve `registry.py` | `models/` | net -60 | models-consolidation.md §5 | pending |
| 3.6 | Inline `_training.py` single-use utilities | `models/` | net -19 | models-consolidation.md §6 | pending |
| 3.7 | `temporal.py` checkpoint fix | `models/temporal.py` | -6 | models-consolidation.md §7 | pending |

## Tier 4: When writing the paper

| # | What | File(s) | Est. delta | Plan reference | Status |
|---|------|---------|------------|----------------|--------|
| 4.1 | ~~Artifacts rewrite~~ | ~~artifacts/~~ | ~~net -220~~ | ~~artifacts-consolidation.md~~ | **DONE** (`analyzer.py`, 2026-03-28) |
| 4.2 | DQN/Bandit -> LightningModules | `models/` | net -120 | models-consolidation.md §8 | pending |
| 4.3 | Memory bloat spike (prefetch thread) | spike | experimental | memory-profiling/performance-analysis.md | pending |

## Completed tiers

| What | Completed | Details |
|------|-----------|---------|
| Config flatten (Hydra -> jsonargparse) | 2026-03-28 | `plans/architecture/flatten-model-config.md` |
| Artifacts `analyze` subcommand | 2026-03-28 | `graphids/core/artifacts/analyzer.py` |
| Lightning callback extraction + LightningCLI | 2026-03-27 | `graphids/cli.py` (GraphIDSCLI) |
| Config system rewrite | 2026-03-26 | `graphids/config/` |
| Pipeline deletion (runner, stages, manifest) | 2026-03-26 | `graphids/pipeline/` deleted entirely |
| Codebase cleanup (PyG APIs, Lightning built-ins) | 2026-03-25 | Custom DataLoader/collation replaced |

## Cross-references

| Plan file | Scope | Status |
|-----------|-------|--------|
| `architecture/models-consolidation.md` | 13 model files, net -284 lines | proposed |
| `architecture/preprocessing-consolidation.md` | 8 data files, -179 lines | proposed |
| `experiment-sweep-plan.md` | Ablation + HPO design (claims/configs) | active |
| `architecture/flatten-model-config.md` | Config flatten reference | completed |
