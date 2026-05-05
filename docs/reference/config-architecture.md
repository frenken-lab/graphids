# Config Architecture

> Jsonnet composition -> Pydantic validation -> direct instantiation.
> For merge semantics, null preservation, env vars, path scheme, and
> observability wiring, see `.claude/rules/config-system.md`.

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

Multi-stage chains (e.g. `autoencoder → supervised → fusion`) come from
a *plan* — a jsonnet file declaring `{ nodes: [...] }`. Shipped plans
live under `configs/plans/`; `configs/plans/ofat.jsonnet` is the OFAT
topology. `python -m graphids run <plan.jsonnet> --dataset X --seed N
--cluster C` walks the plan in topological order via submitit,
threading each node's jid into downstream deps' `afterok`. FINISHED
nodes are skipped via an MLflow check before submission; `--force`
overrides. `python -m graphids status <plan.jsonnet>` queries MLflow
per node. No bash manifest, no parser — the plan jsonnet IS the
artifact.

### Route B: Operational commands (no training)

```
python -m graphids {analyze|rebuild-caches|compare}
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
| `config/catalog.py` | Dataset catalog (`load_catalog`, `dataset_names`), path helpers (`data_dir`, `cache_dir`) | No |
| `orchestrate/config.py` | `ResolvedConfig`, `InstantiatedRun` | No |
| `orchestrate/stage.py` | `build`, `train`, `evaluate` primitives | Yes |
| `core/artifacts/analyzer.py` | `Analyzer(spec)` — invoked by `analyze` blueprint action via `orchestrate.analyze` | Yes |
| `core/monitoring.py` | `SlurmResourceDetector` (OTel resource attrs) | No |
| `core/mlflow_callback.py` | `MLflowTrainingCallback` (per-epoch metrics + finalize) | Yes |
| `_mlflow.py` | `start_training_run`, `log_epoch_metrics`, `log_test_run`, lifecycle | Lazy |
| `_otel.py` | `init_providers`, `wire_file_exporters` | No |

---

## 5. File Layout

```
configs/
├── _lib/defaults.libsonnet        # trainer / checkpoint / early_stopping defaults
├── ablations/{unsupervised,fusion,gat_sampling,gat_loss,id_encoding}/*.jsonnet
├── stages/{autoencoder,supervised,fusion}.jsonnet
├── models/
│   ├── {supervised,unsupervised,fusion}.libsonnet
│   └── fusion/{base,reward}.libsonnet + fusion/methods/*.libsonnet
├── plans/ofat.jsonnet             # multi-stage DAG topology
├── datasets/dataset_registry.json
├── matrix/{axes,topology}.json    # valid model types / stage existence
└── resources/submit_profiles.json # raw parsl SlurmProvider kwargs, [mode][cluster][length]
```

`graphids/` package layout: see `ls graphids/` — every name is self-describing.
The non-obvious ones: `orchestrate.py` is a single module (not a subpackage)
holding `ResolvedConfig`, `InstantiatedRun`, `build_run`, `build`, `train`,
`evaluate`. `_mlflow.py` owns the entire MLflow surface (run lifecycle,
search filter, logged-model registration, dataset lineage).

---

## 6. Running

```bash
# Local dev — renders defaults, trains to run_dir from jsonnet
python -m graphids fit --config configs/stages/autoencoder.jsonnet

# Override via TLA
python -m graphids fit \
    --tla 'dataset="hcrl_sa"' \
    --tla 'scale="large"' \
    --tla 'variational=false' \
    --config configs/stages/autoencoder.jsonnet \
    --set model.init_args.lr=0.005

# SLURM ablation
python -m graphids submit configs/ablations/unsupervised/vgae.jsonnet --dataset set_01 --seed 42
```

---

## 7. Stage / Ablation Function Convention

Every `stages/*.jsonnet` and `ablations/**/*.jsonnet` is a top-level
function with sensible defaults for every TLA. Adding a new TLA means
updating the jsonnet signature + (if the TLA is launcher-level) the
matching flat flag in `graphids/cli/submit.py` (which appends to the
inline `flag_tlas` list — there is no separate helper).

```jsonnet
// Stage (configs/stages/*.jsonnet) — no overrides TLAs.
function(
  dataset='hcrl_ch', seed=42, run_dir='',
  scale='small', conv_type='gatv2', variational=true,
  auxiliaries=[], vgae_ckpt_path=null,
  ckpt_path=null,
)
  defaults.trainer + defaults.checkpoint + defaults.early_stopping
  + vgae.base + vgae.scales[scale]
  + { seed_everything: seed, trainer+: {...}, data: {...}, model+: {...} }

// Ablation preset (configs/ablations/**/*.jsonnet) — wraps stage in mergePatch.
function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, conv_type=pd.conv_type,
  ckpt_path=null,
)
  std.mergePatch(
    stage(
      dataset=dataset, seed=seed, scale=scale,
      run_dir=std.native('paths.run_dir')(dataset, 'unsupervised', 'vgae', seed),
      conv_type=conv_type, model_type='vgae', variational=true,
      ckpt_path=ckpt_path,
    ) + { trainer+: { max_epochs: 1200 } },  // group defaults as nested obj
    std.extVar('overrides'),                 // user --set flags
  )
```

---

## 8. Robustness

1. **Typed TLA round-trip.** `render` passes every TLA through
   `--tla-code <k>=<json.dumps(v)>` so ints stay ints, bools stay bools,
   lists stay lists, `None` becomes jsonnet `null`.
2. **Pydantic gate (`ValidatedConfig`)** — null list fields in
   `model.init_args`, monitor mismatch between `checkpoint` and
   `early_stopping`, un-namespaced `class_path` strings, and
   `LearningRateMonitor` without `logger` all die with an actionable
   error before any torch import.
3. **Signature-filtered kwargs** — `build_run` runs every class_path's
   `init_args` through `filter_kwargs(klass, init_args)` so jsonnet can
   pass fields the target class doesn't accept without raising.
4. **`topology.py` import-time assertions** — every model family has a
   libsonnet, every stage has a `.jsonnet`, every fusion method has a
   method libsonnet; `submit_profiles.json` `scale_mult` keys are in
   `VALID_SCALES`. Missing files / bad keys fail at package import.
