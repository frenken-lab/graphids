# Orchestration — `graphids/orchestrate.py`

> Status: **implemented** | Last refactor: 2026-05-04 (jsonnet → Python
> plan layer; Parsl `SlurmProvider` replaces submitit)

A training run is a Python plan's `build()` output → `BlueprintArray`
validation → `run_row(row)`. No planner, no cross-stage driver
in-process. Multi-stage chains are declared in a Python plan
(`graphids/configs/plans/<name>.py`); `graphids run <name>` emits a
JSON array, the user/LLM iterates and submits per row — see
`submit-flow.md`.

## Layout

`orchestrate.py` is a single module (not a subpackage). Public surface:

| Function | Role |
|---|---|
| `run_row(row, *, ckpt_path=None)` | Top-level dispatch on `row.action`. The only entry called by `graphids exec` and by the SLURM job's `srun` line. |
| `_instantiate(spec)` | Recursively builds `{class_path, init_args}` blocks via importlib + `getattr`; sub-dicts with their own `class_path` are built bottom-up. |

## Execution flow

```
graphids exec --row '<json>'
  +-- BlueprintArray.model_validate([row])  → typed Row (discriminated union)
  +-- run_row(row, ckpt_path=...)
        +-- match row.action:
              fit / test  → instantiate trainer/model/datamodule;
                            open MLflow run; trainer.fit(...) / trainer.test(...)
              extract     → write fusion features cache (idempotent on output_dir)
              analyze     → core.artifacts.Analyzer(row) → per-checkpoint artifacts
              cmd         → run shell command
```

## Key decisions

| Decision | Rationale |
|---|---|
| Path math is one Python module | `graphids.config.catalog` defines `run_dir` / `best_ckpt` / `states_dir`. Plans import directly via `graphids.configs.catalog`. No native-callback bridge — single source. |
| No in-process multi-stage driver | A Python plan declares the topology; `graphids run` emits JSON, the user/LLM iterates `graphids submit --row "$row"`. No scheduler re-query. See `submit-flow.md`. |
| `run_row` is the single dispatch | One entry from CLI, one from the SLURM job. No `fit`/`test` Typer commands; pipeline shape is `run | exec | submit`. |
