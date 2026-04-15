# Orchestration — `graphids/orchestrate/`

> Status: **implemented** | Last refactor: 2026-04-15 (pipeline route
> deleted — one route, ablation presets own their own run_dir)

A training run is a jsonnet preset rendered to a dict, validated, then
fed through `build → train → evaluate`. No planner, no cross-stage
driver. Multi-stage chains are built in bash by submitting each preset
with `SBATCH_DEP=afterok:<jid>` between them.

## Layout

| Module | Role |
|---|---|
| `config.py` | `ResolvedConfig`, `InstantiatedRun` — boundary types. |
| `instantiate.py` | `build_run(rendered)` — class_path + signature-filtered kwargs, callback/logger wiring. |
| `stage.py` | `build(resolved)`, `train(artifacts, resolved)`, `evaluate(artifacts, resolved)`. |

## Execution flow

```
fit | test  (cli/training.py)
|
+-- render(config_path, tla=...)              [config/jsonnet.py]
+-- apply_overrides(rendered, --set ...)      [cli/app.py]
+-- ResolvedConfig.from_rendered(rendered)    [orchestrate/config.py]
|     -> validate_config(...)                 [config/schemas.py]
|     -> run_dir = trainer.default_root_dir
+-- build(resolved)                           [stage.py]
|     -> gc + torch.cuda reset
|     -> build_run(rendered, validated)       [instantiate.py]
+-- train(artifacts, resolved, resume_from)   [stage.py]
|     -> wire_file_exporters(run_dir)
|     -> trainer.fit(...)
|     -> touch .train_complete
+-- evaluate(artifacts, resolved)             [stage.py]
      -> trainer.test(...)
      -> touch .test_complete + save predictions
```

## Key decisions

| Decision | Rationale |
|---|---|
| Jsonnet preset owns `run_dir` | Every `configs/ablations/*.jsonnet` computes `run_dir` from `(lake_root, dataset, seed)` via `_paths.libsonnet`. Path logic lives next to the config, not in a Python planner. |
| No in-process multi-stage driver | A bash loop over `scripts/run <preset>` with `afterok` deps does this without a parallel Python declaration. |
| `build` / `train` / `evaluate` are dumb primitives | No `ResolvedConfig` knowledge, no cache knowledge. Same primitives used by `fit` and `test`. |
