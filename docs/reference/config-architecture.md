# Config Architecture

> Jsonnet composition -> Pydantic validation -> direct instantiation.
> For file layout, stage conventions, and running examples, see `.claude/rules/config-system.md`.

---

## 1. CLI Routes

Three routes end in training, plus operational commands:

### Route A: Dev CLI (interactive)

```
python -m graphids fit \
    --tla 'dataset="hcrl_ch"' \
    --tla 'scale="small"' \
    --config configs/stages/autoencoder.jsonnet \
    --model.init_args.lr=0.01
  -> __main__.py
  -> cli._training (Typer @app.command)
  -> render_config(jsonnet_path, tla)
  -> validate_config(rendered)  # Pydantic gate
  -> _apply_overrides(merged, overrides)
  -> instantiate(merged, validated=...)
  -> trainer.fit(model, datamodule=datamodule, ckpt_path=...)
```

### Route B: Pipeline (Monarch -> SLURM -> in-process)

```
python -m graphids monarch-run --dataset hcrl_sa --seed 42 --scale small
  -> cli/_monarch.py (Typer @app.command)
  -> PipelineConfig(**kwargs) -> TrainingRunConfig
  -> build_pipeline_stages(config) -> list[StageConfig]
     +- enumerate_assets(recipe)
  -> JobSpec.create_job() -> monarch SlurmJob
     +- patch_clusterscope_for_osc()
  -> run_chain(stages, spec, dataset=..., seed=..., max_retries=...)
    -> job.state().pipeline.spawn_procs(per_host={"gpus": N})
    -> PipelineActor spawned on proc_mesh:
        train_stage(stage_config, dataset, seed, upstream_ckpts)
          -> ResolvedConfig.resolve(cfg, lake_root, user, dataset, seed, upstream_ckpts)
              +- PathContext(...)
              +- _build_tla_dict(...)                (private, resolve.py)
              +- render(jsonnet_path, tla)           (config/jsonnet.py)
              +- validate_config(rendered)           (config/schemas.py)
              +- monitor/mode consistency check      (inline log warning)
          -> instantiate(resolved.rendered, validated=resolved.validated)
          -> trainer.fit(model, datamodule)
```

### Route C: Validation (resolver gate)

Validation runs inside `ResolvedConfig.resolve()`:
`render(...)` -> `validate_config(rendered)` -> inline monitor/mode consistency check (log warning on mismatch).

### Route D: Operational commands (no training)

```
python -m graphids {analyze|profile|rebuild-caches|stage-data|...}
  -> __main__.py imports cli submodules
  -> Typer @app.command() dispatch per submodule
```

**Key invariant:** Routes A and B render configs through the same
`graphids.config.jsonnet.render_config` shim. One composition primitive,
one subprocess call to `go-jsonnet`.

---

## 2. Pydantic Validation Layer

`graphids/config/schemas.py::validate_config(rendered) -> ValidatedConfig`
is the structural gate that runs **immediately after** `render_config` on
every path. Torch-free, deterministic.

### Schema tree

```
ValidatedConfig (extra="forbid")
+-- seed_everything: int
+-- trainer: TrainerSection    (extra="allow" -- Lightning Trainer has ~50 kwargs)
+-- data: ClassPathBlock       (extra="forbid"; class_path required)
+-- model: ClassPathBlock      (extra="forbid"; class_path required)
+-- checkpoint: CheckpointSection  (mode: Literal["min","max"])
+-- early_stopping: EarlyStoppingSection  (mode: Literal["min","max"])
+-- ckpt_path: str | None      (auto-resume passthrough)
```

### Model validators

| Validator | Rule | Why it exists |
|---|---|---|
| `_no_null_list_fields` | `model.init_args.{pool_aggrs, hidden_dims, auxiliaries}` must not be null | jsonargparse rejects these at instantiation with a cryptic error |
| `_monitor_pair_consistent` | `checkpoint.monitor/mode == early_stopping.monitor/mode` | Divergent monitors = typo in the stage libsonnet |
| `_lr_monitor_requires_logger` | `LearningRateMonitor` callback needs `trainer.logger != False` | LR monitor is silently disabled without a logger |
| `_class_paths_namespaced` | `data.class_path` and `model.class_path` must start with `graphids.` | Catches relative imports and stray modules |

Stage-archetype monitor mismatches (fusion must be `val_acc/max`, every
other stage `val_loss/min`) are a **warning** in `orchestrate/resolve.py`
because they're advisory, not fatal.

### Integration points

| Call site | What it does |
|---|---|
| `ResolvedConfig.resolve()` | Calls `validate_config(rendered)` after `render_config`; attaches typed view to `ResolvedConfig.validated` |
| `instantiate()` | Re-validates if caller didn't pass a `ValidatedConfig` |

---

## 3. Forced Callbacks + Direct Instantiation

Critical callbacks are protected by living at top-level namespaces in the
rendered dict — `checkpoint.*`, `early_stopping.*`, etc. — and being
constructed explicitly by `instantiate._build_callbacks()`. Any stage-level
`trainer.callbacks` appends user callbacks; it cannot drop the forced set.

Forced callbacks (from `defaults.libsonnet`): ModelCheckpoint, EarlyStopping,
OTelTrainingCallback. Logger: OTelTrainingLogger.

### instantiate() responsibilities

`graphids.instantiate.instantiate(rendered, validated=None)`:

| Step | How |
|---|---|
| Class-path import | `importlib.import_module` + `getattr` |
| link_arguments | `_apply_link_arguments(merged, dm_cls, model_cls)` — signature-filtered |
| Forced callbacks | `_build_callbacks(merged, default_root_dir)` — explicit construction |
| Path patching | inline in `_build_callbacks` (checkpoint dirpath) and `_build_loggers` (logger save_dir) |
| KD loss injection | `inject_loss_fn` pops `distillation_config`, builds loss via `build_loss()` |
| seed_everything | explicit `graphids.core.trainer.seed_everything(merged["seed_everything"])` |

---

## 4. Key Files

| File | Role | Torch? |
|---|---|---|
| `cli/_training.py` | Dev-path Typer entry — `fit/test/validate/predict`, `--config`, `--tla`, `--ckpt_path` | Lazy |
| `cli/_monarch.py` | `monarch-run`, `monarch-sweep` commands | Lazy |
| `instantiate.py` | `instantiate(rendered) -> InstantiatedRun` — importlib, link_arguments, forced callbacks | Yes |
| `__main__.py` | Imports `cli/` submodules to register Typer commands; OTel Phase A init | Lazy |
| `config/jsonnet.py` | `render_config(path, tla)` subprocess shim | No |
| `config/schemas.py` | `ValidatedConfig`, `validate_config`, `ConfigValidationError` | No |
| `config/topology.py` | Stage DAG, valid types/scales, import-time assertions | No |
| `orchestrate/planning/recipes.py` | `TrainingRunConfig`, `KDEntry` | No |
| `orchestrate/planning/planner.py` | `StageConfig`, `enumerate_assets` | No |
| `orchestrate/resolve.py` | `ResolvedConfig.resolve` — builds TLA, renders, validates, cross-field checks | No |
| `orchestrate/run.py` | `PipelineConfig`, `build_pipeline_stages`, `run_pipeline` driver | No |
| `orchestrate/allocate.py` | `JobSpec`, `build_slurm_job`, `spawn_actor` | No |
| `orchestrate/chain.py` | `run_chain(actor, stages, …) → ChainResult` | No |
| `orchestrate/stage.py` | `build`, `train`, `evaluate`, `run_stage` primitives | Yes |
| `orchestrate/analyze.py` | pipeline-level `analyze` + `run_single_analysis` helper | Yes |
| `orchestrate/actors.py` | `PipelineActor` — thin Monarch endpoint wrapper (`train_stage` / `eval_stage` / `analyze_stage`) | Yes |
| `core/monitoring.py` | `OTelTrainingCallback`, `OTelTrainingLogger` | Yes |
| `core/otel.py` | `init_providers`, `wire_file_exporters` | No |

---

## 5. Architecture Evaluation

### Strengths

| # | Strength |
|---|---|
| S1 | **Single composition primitive** — jsonnet replaces custom deep-merge + dotted-override + stringification |
| S2 | **Torch-free config boundary** — jsonnet.py, schemas.py, resolve.py never import torch |
| S3 | **Typed TLA round-trip** — ints stay ints, bools stay bools via `--tla-code` JSON encoding |
| S4 | **Single convergence point** — every path ends at `instantiate(rendered, validated=...)` |
| S5 | **Forced callbacks via explicit construction** — stage jsonnets can add but never drop critical callbacks |
| S6 | **Import-time config validation** — `topology.py` cross-validates jsonnet tree at package import |
| S7 | **Pydantic `extra="forbid"`** — typos caught at construction time |
| S8 | **Content-addressed run dirs** — deterministic, filesystem-navigable, resumable |

### Known limitations

| # | Issue | Severity |
|---|---|---|
| L1 | jsonnet rendering shells out per-render (~5 ms subprocess cost) | Low |
| L2 | `jsonargparse` remains only in `cli/_analysis.py` (Analyzer config) | Low |
| L3 | Fusion stage absorbs unused TLAs (`auxiliaries`, `vgae_ckpt_path`) | Low |
