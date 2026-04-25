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
| Path scheme is one Python module | `graphids.config.paths` defines `run_dir`/`vgae_ckpt`/`states_dir`. Jsonnet calls into it via `std.native('paths.run_dir')(...)` — registered as `native_callbacks` in `render()`. `slurm/dag.py` imports the same module. No parallel jsonnet implementation. |
| No in-process multi-stage driver | An in-memory DAG (`graphids.slurm.dag.OFAT_DAG`) calling `graphids.slurm.submit.submit()` with `dep_jids` holds jids between stages; no scheduler re-query (kills the Stage 3 dep-race). |
| `build` / `train` / `evaluate` are dumb primitives | No `ResolvedConfig` knowledge, no cache knowledge. Same primitives used by `fit` and `test`. |
