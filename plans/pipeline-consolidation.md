# Pipeline Consolidation Plan

> Status: **superseded** | Original: 2026-03-27 | Updated: 2026-03-28

## What happened

This plan proposed replacing scattered orchestration code with a thin DAG orchestrator
and Lightning-native job configuration. **Most of this was executed**, but through a
more aggressive path than planned: the entire `graphids/pipeline/` package was deleted
rather than incrementally consolidated.

Key commits: `95fa998`, `c41511d`, `ef16645` (2026-03-26--27).

## Current state vs plan

### Layer 1: Within a job -- DONE

| Planned | Actual |
|---------|--------|
| One YAML config per stage | Yes -- `graphids/config/stages/*.yaml` (7 stage configs) |
| LightningCLI reads config | Yes -- `GraphIDSCLI` in `__main__.py` |
| `SLURMEnvironment(auto_requeue=true)` | Wired in `trainer.yaml` |
| `ModelCheckpoint(save_last=true)` | Wired in `trainer.yaml` |
| `DeviceStatsMonitor` | Wired in `trainer.yaml` |
| No custom Python runner | Yes -- `runner.py` deleted |

### `_preamble.sh` and `_epilog.sh` -- TRIMMED

| Script | Plan target | Actual |
|--------|-------------|--------|
| `_preamble.sh` | ~10 lines | 37 lines (still has data staging + CUDA config) |
| `_epilog.sh` | ~15 lines | 15 lines (done) |

`_preamble.sh` is larger than planned because `prepare_data()` hasn't been wired into
DataModules yet (preprocessing-consolidation.md S6a). Data staging still happens in shell.

### Layer 2: Across jobs -- PARTIALLY DONE

| Planned | Actual |
|---------|--------|
| `graphids/orchestrate/` package | Exists: `__init__.py`, `__main__.py`, `resources.py`, `submit.py` |
| DAG topology from `pipeline.yaml` | Yes -- `graphids/config/__init__.py` exports `STAGE_DEPENDENCIES` |
| Adaptive retry (OOM -> 2x mem) | In `resources.py` |
| Config generation for ablations | Not yet -- ablation configs still hand-written |
| Dagster UI optional viz | Not pursued |

### Deleted files -- ALL DONE

| Planned deletion | Status |
|---|---|
| `graphids/pipeline/` (entire package) | Deleted |
| `cluster.py` | Deleted (was in pipeline/) |
| `graphids/pipeline/callbacks.py` | Deleted |
| `graphids/pipeline/cli.py` | Deleted |
| `graphids/pipeline/manifest.py` | Deleted |
| `graphids/pipeline/stages/` (9+ files) | Deleted |

### Spike test results (preserved)

Job `46012629` (2026-03-27): `SLURMEnvironment` detected SLURM, `ModelCheckpoint` wrote
both `best_model.ckpt` and `last.ckpt`, `DeviceStatsMonitor` ran. Core Lightning + SLURM
contract verified on Pitzer.

## Remaining items

1. **`graphids/__init__.py` stale lazy import**: Still has `"pipeline"` in `_lazy_submodules`
   but `graphids/pipeline/` doesn't exist. Will fail at runtime on `import graphids.pipeline`.
2. **Config generation** (`generate_configs.py`): Ablation configs are hand-written YAML.
   The planned generator (ablation spec -> per-stage YAML) hasn't been built.
3. **`_preamble.sh` further trim**: Needs `prepare_data()` in DataModules first
   (blocked on preprocessing-consolidation.md S6a).

## Dependencies on other plans

| This plan needs | From plan | Status |
|---|---|---|
| `DataModule.prepare_data()` for data staging | Preprocessing S6a | pending |
| `configure_optimizers` deleted from modules | Models S1 | pending |
| Stage YAML configs referencing correct class paths | Already done | done |

---

<details>
<summary>Original plan details (2026-03-27)</summary>

The original plan included detailed architecture for:
- Full YAML-only job configuration (implemented)
- Dagster-as-library orchestrator (not pursued; simpler submit.py built instead)
- Interactive CPU job poll loop (implemented in submit.py)
- `fire_and_forget` sbatch chaining (implemented)
- Resource profiles + failure reactions (implemented in resources.py)

See git history for the original 600-line plan.
</details>
