# KD_GAT_RECIPE env var not picked up by dagster — RESOLVED

## Root Cause

`component.py:35` read `KD_GAT_RECIPE` at **module import time**:
```python
RECIPE_PATH = Path(os.environ.get("KD_GAT_RECIPE", RECIPES_DIR / "ablation.yaml"))
```

`dg launch` imports `graphids.orchestrate.definitions` → imports `component.py` → `RECIPE_PATH` frozen at first import. The dagster multiprocess executor builds the execution plan (which assets to materialize) in the **parent process** at `dagster/_core/executor/multiprocess.py:209`. Plan is frozen before child workers spawn. If `RECIPE_PATH` resolved to `ablation.yaml` during import, all 22 ablation assets got planned regardless of the env var being correct later.

## Chain of Events

1. `KD_GAT_RECIPE=smoke_test.yaml scripts/submit.sh ablation ...` — env var set in calling shell
2. sbatch inherits env, submits SLURM job 46256266 on CPU partition
3. SLURM runs `--wrap` string: `SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 source _preamble.sh && dg launch ...`
4. `_preamble.sh` runs `set -a; source .env; set +a` — `.env` doesn't touch `KD_GAT_RECIPE`
5. `_preamble.sh` line 27 runs `python -c "from graphids.config import WANDB_WRITE_DIR; ..."` — this imports `graphids.config` but NOT `component.py`
6. `dg launch` starts, loads code location by importing `graphids.orchestrate.definitions`
7. `definitions.py:11` imports `SlurmTrainingComponent` from `component.py`
8. `component.py:35`: `RECIPE_PATH = Path(os.environ.get("KD_GAT_RECIPE", ...))` — **this is where the env var is read**
9. If env var was somehow not propagated to the `dg launch` process at this point, falls back to `ablation.yaml`
10. `definitions.py:17`: `defs = build_defs_for_component(component)` → calls `build_defs()` → reads frozen `RECIPE_PATH` → expands ablation recipe → 22 assets
11. `execute_materialize_command` builds execution plan from these 22 assets
12. Multiprocess executor dispatches steps from frozen plan — 5 `normal_*` GPU jobs submitted

## Why isolated tests passed

All tests set `KD_GAT_RECIPE` before importing `component.py` in a fresh Python process. The real failure mode is that `RECIPE_PATH` is evaluated once at import time and never re-read. In the real `dg launch` flow, the import timing and env var availability didn't align.

## Fix Applied

Moved env var read from module scope to `build_defs()` body (runtime, not import time):
```python
def build_defs(self, context):
    recipe_path = Path(os.environ.get("KD_GAT_RECIPE", RECIPES_DIR / "ablation.yaml"))
    recipe = expand_recipe_configs(read_yaml(recipe_path))
```

## Prevention Rule

Never read env vars that determine pipeline topology (asset count, recipe selection) at module scope. Module-level code runs on every import — timing is not guaranteed. Use `build_defs()` for pipeline-shaping decisions, `@asset` function body for per-asset decisions.

## Collateral

- Job 46256266: orchestrator ran ablation recipe, submitted 5 `normal_*` GPU jobs (~25 min V100 time wasted)
- 5 multiprocess executor child jobs (46256280-84) failed with "no --partition" error (separate issue)
- Analysis assets ran in-process on CPU node without GPU (separate issue: analysis-assets-in-process.md)
