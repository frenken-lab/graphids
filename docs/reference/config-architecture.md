# Config Architecture

> Jsonnet composition -> Pydantic validation -> direct instantiation.
> For file layout, stage conventions, and running examples, see `.claude/rules/config-system.md`.

---

## 1. CLI Routes

One training route + operational commands:

### Route A: Train a preset

```
python -m graphids fit \
    --tla 'dataset="hcrl_ch"' \
    --tla 'scale="small"' \
    --config configs/ablations/unsupervised/vgae.jsonnet \
    --set model.init_args.lr=0.01
  -> __main__.py
  -> cli.training (Typer @app.command)
  -> dotted_to_nested(--set ...)               # cli/app.py
  -> render(jsonnet_path, tla, set_overrides)  # passes overrides as ext_code
                                               # registers paths.* native_callbacks
  -> ResolvedConfig.from_rendered(rendered)    # validates + pulls run_dir
  -> build(resolved)  ->  train(artifacts, resolved, resume_from=--ckpt-path)
```

Every ablation preset under `configs/ablations/*.jsonnet` computes its
own `run_dir` via `std.native('paths.run_dir')(dataset, group, variant,
seed)` — `render()` registers `graphids.config.paths.run_dir` (and
`vgae_ckpt`, `states_dir`) as jsonnet native callbacks so both
languages share one path scheme. `run_root` flows in via
`std.extVar('run_root')` from `GRAPHIDS_RUN_ROOT` (per-user, distinct
from `LAKE_ROOT`). User `--set` flags apply via `std.mergePatch` at
each preset's apex.

The SLURM submitter (`python -m graphids submit`, library:
`graphids.slurm.submit.submit()`) just forwards TLAs.

Multi-stage chains (e.g. `autoencoder → supervised → fusion`) are a
Python DAG driver (`graphids.slurm.dag`, CLI `python -m graphids
launch-ablation`). Topology is a declarative `OFAT_DAG` tuple of
`FitNode`/`ExtractStatesNode`; the executor walks it in topological
order, calling `graphids.slurm.submit.submit()` directly with `dep_jids`
derived from the in-memory jid map. There is no subprocess and no
in-process pipeline driver.

### Route B: Operational commands (no training)

```
python -m graphids {analyze|rebuild-caches|extract-fusion-states|compare}
  -> __main__.py imports cli submodules
  -> Typer @app.command() dispatch per submodule
```

---

## 2. Pydantic Validation Layer

`graphids/config/schemas.py::validate_config(rendered) -> ValidatedConfig`
runs immediately after `render` on every path. Torch-free, deterministic.

### Schema tree

```
ValidatedConfig (extra="forbid")
+-- seed_everything: int
+-- trainer: TrainerSection    (extra="allow" -- TrainerConfig dataclass kwargs flow through)
+-- data: ClassPathBlock       (extra="forbid"; class_path required)
+-- model: ClassPathBlock      (extra="forbid"; class_path required)
+-- checkpoint: CheckpointSection  (mode: Literal["min","max"])
+-- early_stopping: EarlyStoppingSection  (mode: Literal["min","max"])
+-- ckpt_path: str | None      (auto-resume passthrough)
```

### Model validators

| Validator | Rule | Why it exists |
|---|---|---|
| `_no_null_list_fields` | `model.init_args.{pool_aggrs, hidden_dims, auxiliaries}` must not be null | Instantiation rejects null lists with a cryptic error |
| `_monitor_pair_consistent` | `checkpoint.monitor/mode == early_stopping.monitor/mode` | Divergent monitors = typo in the stage libsonnet |
| `_lr_monitor_requires_logger` | `LearningRateMonitor` callback needs `trainer.logger != False` | LR monitor is silently disabled without a logger |
| `_class_paths_namespaced` | `data.class_path` and `model.class_path` must start with `graphids.` | Catches relative imports and stray modules |

---

## 3. Forced Callbacks + Direct Instantiation

Critical callbacks are constructed explicitly by
`instantiate._build_callbacks()`. Any stage-level `trainer.callbacks`
appends user callbacks; it cannot drop the forced set.

Forced callbacks (from `defaults.libsonnet`): ModelCheckpoint,
EarlyStopping, MLflowTrainingCallback, CurriculumEpochCallback. No
trainer logger (MLflow callback handles metrics).

### build_run() responsibilities

`graphids.orchestrate.instantiate.build_run(rendered, validated=None)`:

| Step | How |
|---|---|
| Class-path import | `importlib.import_module` + `getattr` |
| Signature-filtered kwargs | `filter_kwargs(klass, init_args)` |
| Callbacks / logger | `build_callbacks(rendered)` / `build_loggers(rendered)` — explicit construction |
| KD loss injection | `inject_loss_fn` pops `distillation_config`, builds loss via `build_loss()` |
| seed_everything | explicit `seed_everything(rendered["seed_everything"])` |

---

## 4. Key Files

| File | Role | Torch? |
|---|---|---|
| `cli/training.py` | `fit` / `test` — renders preset, builds + runs | Lazy |
| `cli/app.py` | Typer root + `dotted_to_nested` for `--set` | No |
| `instantiate.py` | `build_run(rendered) -> InstantiatedRun` — importlib, filter_kwargs, callback wiring | Yes |
| `__main__.py` | Imports `cli/` submodules to register Typer commands | Lazy |
| `config/jsonnet.py` | `render(path, tla, set_overrides)` — passes `run_root` ext_code + `paths.*` native_callbacks | No |
| `config/paths.py` | Canonical `run_dir` / `vgae_ckpt` / `states_dir` scheme (shared with jsonnet) | No |
| `config/settings.py` | `GraphIDSSettings` — pydantic-settings, auto-loads `./.env` | No |
| `config/schemas.py` | `ValidatedConfig`, `validate_config` | No |
| `config/topology.py` | Stage-file existence check, dataset catalog | No |
| `orchestrate/config.py` | `ResolvedConfig`, `InstantiatedRun` | No |
| `orchestrate/stage.py` | `build`, `train`, `evaluate` primitives | Yes |
| `core/analysis/runner.py` | `run_single_analysis` — invoked by `graphids analyze` CLI | Yes |
| `core/monitoring.py` | `SlurmResourceDetector` (OTel resource attrs) | No |
| `core/mlflow_callback.py` | `MLflowTrainingCallback` (per-epoch metrics + finalize) | Yes |
| `_mlflow.py` | `start_training_run`, `log_epoch_metrics`, `log_test_run`, lifecycle | Lazy |
| `_otel.py` | `init_providers`, `wire_file_exporters` | No |
