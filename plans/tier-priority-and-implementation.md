# Priority Tier List + Implementation Details

> Status: **active** | Date: 2026-03-28

---

## Tier 3: Code consolidation (when you have a clean week)

| #   | What                                       | File(s)                   | Lines   | Details in                                   |
| --- | ------------------------------------------ | ------------------------- | ------- | -------------------------------------------- |
| 3.1 | Dead code deletion                         | models + preprocessing    | -200    | models-consolidation.md §4, preprocessing §1 |
| 3.2 | `GraphModuleBase` shared base              | `models/_training.py`     | net -90 | models-consolidation.md §2, §3               |
| 3.3 | Delete `configure_optimizers` + wire CLI   | `__main__.py` + 3 modules | net -25 | models-consolidation.md §1                   |
| 3.4 | Preprocessing DataModule conventions       | 3 DataModules             | net +31 | preprocessing-consolidation.md §6            |
| 3.5 | Dissolve `registry.py`                     | `models/`                 | net -73 | models-consolidation.md §6                   |
| 3.6 | Inline `_training.py` single-use utilities | `models/`                 | net -18 | models-consolidation.md §7                   |
| 3.7 | `temporal.py` checkpoint fix               | `models/temporal.py`      | -10     | models-consolidation.md §8                   |

## Tier 4: When writing the paper

| #   | What                                                | File(s)           | Lines        | Details in                                |
| --- | --------------------------------------------------- | ----------------- | ------------ | ----------------------------------------- |
| 4.1 | Artifacts rewrite (embeddings, CKA, loss landscape) | `core/artifacts/` | net -220     | artifacts-consolidation.md                |
| 4.2 | DQN/Bandit → LightningModules                       | `models/`         | net -120     | models-consolidation.md §5                |
| 4.3 | Memory bloat spike (prefetch thread)                | spike             | experimental | memory_profiling/resource_plan §Problem 2 |

## Cross-references

| Plan file                                      | Scope                                  |
| ---------------------------------------------- | -------------------------------------- |
| `plans/models-consolidation.md`                | 13 model files, -287 lines             |
| `plans/preprocessing-consolidation.md`         | 8 data files, -179 lines               |
| `plans/artifacts-consolidation.md`             | 6 artifact files, -220 lines           |
| `plans/pipeline-consolidation.md`              | Orchestration + SLURM                  |
| `memory_profiling/resource_plan_2026_03_27.md` | Resource profiles + collation analysis |
