# Config Architecture

> CLI routes, jsonnet-based config composition, and architecture evaluation.
> Phase 1 migration (2026-04-05) replaced the 3-chain YAML + `merge_yaml_chain`
> plumbing with `render_config(jsonnet_path, tla)`. Phase 2 (2026-04-05)
> added a Pydantic validation layer — `graphids.config.validate_config` —
> between the render and the downstream consumer. Phase 3 (2026-04-05)
> stripped `LightningCLI` entirely: the validated dict is now consumed
> by `graphids.core.instantiate.instantiate`, which imports class_paths
> directly via `importlib` and constructs the Lightning stack without
> `GraphIDSCLI`. Phase 4 retools the analyzer CLI to keep jsonargparse
> for Jsonnet-backed configs (`commands/analyze.py`).

---

## 1. CLI Routes

Three routes end in training, plus operational commands:

### Route A: Dev CLI (interactive)

```
python -m graphids fit \
    --tla 'dataset="hcrl_ch"' \
    --tla 'scale="small"' \
    --config configs/stages/autoencoder.jsonnet \
    --set model.init_args.lr=0.01
  -> __main__.py
  -> commands.train.main([subcommand, *argv])
  -> argparse: --config, --tla, --set, --ckpt_path
  -> render_config(jsonnet_path, tla)
  -> validate_config(rendered)  # Pydantic gate
  -> _apply_set_overrides(merged, overrides)
  -> instantiate(merged, validated=...)
       # importlib → class_path; signature-filtered link_arguments;
       # forced callbacks; wandb forwarding.
  -> trainer.fit(model, datamodule=datamodule, ckpt_path=...)
```

### Route B: Pipeline (dagster → SLURM → jsonnet → direct instantiate)

```
dg launch --assets '*'
  -> SlurmTrainingComponent.build_defs()
    -> expand_recipe_configs(recipe)
    -> enumerate_assets(PIPELINE_YAML, recipe) -> list[StageConfig]
       # StageConfig.jsonnet_path = "configs/stages/<stage>.jsonnet"
    -> ConfigResolver.resolve(cfg, ...)
       # Builds TrainingSpec(jsonnet_path, jsonnet_tla)
       # render_config → validate_config → cross-field rules
    -> SlurmTrainingResource.submit_and_wait(spec, resources)
      -> sbatch -> SLURM job:

        python -m graphids from-spec --phase train --spec-file /tmp/spec.json
          -> from_spec.main(argv)
          -> run_training_from_spec(spec)
            -> render_config(spec.jsonnet_path, spec.jsonnet_tla)
            -> validate_config(merged)           # belt-and-braces
            -> snapshot_config(merged, run_dir)  # config_snapshot.yaml
            -> instantiate(merged, validated=...)
            -> trainer.fit(model, datamodule)
```

### Route C: Validation (resolver gate)

Validation runs inside `ConfigResolver.resolve` on asset materialization:
`render_config(...)` → `validate_config(rendered)` → `validate_stage_config`
(cross-field rules that depend on ResourceSpec).

### Route D: Operational commands (no LightningCLI)

```
python -m graphids {analyze|landscape|profile|rebuild-caches|stage-data|...}
  -> __main__.py -> _COMMAND_MODULES dispatch
  -> each command has its own argparse + logic
```

**Key invariant:** Routes A, B, and C all render configs through the same
`graphids.config.jsonnet.render_config` shim. One composition primitive,
one subprocess call to `go-jsonnet`.

---

## 2. Config Composition (jsonnet)

### File layout

```
configs/                           # repo root
├── _lib/
│   ├── defaults.libsonnet         # trainer/checkpoint/early_stopping defaults
│   └── helpers.libsonnet          # apply_dotted() for recipe overrides
├── stages/
│   ├── autoencoder.jsonnet        # VGAE + CANBusDataModule
│   ├── normal.jsonnet             # GAT + CANBusDataModule
│   ├── curriculum.jsonnet         # GAT + CurriculumDataModule
│   └── fusion.jsonnet             # fusion-method dispatch
├── models/
│   ├── vgae.libsonnet             # { base, scales: {small, large}, kd }
│   ├── gat.libsonnet
│   └── dgi.libsonnet
├── fusion.libsonnet               # { base, methods: {bandit, dqn, mlp, weighted_avg} }
└── fusion/
    ├── base.libsonnet             # shared fusion trainer + data
    └── methods/
        ├── bandit.libsonnet
        ├── dqn.libsonnet
        ├── mlp.libsonnet
        └── weighted_avg.libsonnet
```

### Stage function shape

Every `stages/*.jsonnet` is a top-level function of TLAs with sensible
defaults for the dev path. `graphids.orchestrate.contracts.build_tla_dict` is
the single mapping between Python-side `StageConfig` and the TLA dict each
stage accepts.

```jsonnet
local defaults = import '../_lib/defaults.libsonnet';
local helpers = import '../_lib/helpers.libsonnet';
local vgae = import '../models/vgae.libsonnet';

function(
  dataset='hcrl_ch', seed=42, run_dir='',
  scale='small', conv_type='gatv2', variational=true,
  auxiliaries=[], vgae_ckpt_path=null,
  trainer_overrides={}, stage_overrides={}, ckpt_path=null,
)
  defaults.trainer + defaults.checkpoint + defaults.early_stopping
  + vgae.base + vgae.scales[scale]
  + {
    seed_everything: seed,
    trainer+: { default_root_dir: run_dir },
    data: { ... dataset: dataset ... },
    model+: { init_args+: { ... } },
  }
  + helpers.apply_dotted(trainer_overrides)
  + helpers.apply_dotted(stage_overrides)
```

### Merge semantics

Jsonnet `+:` is deep-merge; `+` on top-level objects is shallow merge with
last-wins. Lists replace on conflict. This matches the pre-migration
`yaml_utils.deep_merge` exactly — the same layering rules, just expressed
natively in the composition language instead of reimplemented in Python.

**Always use `+:` on nested keys.** A bare `+` on nested dict keys
silently replaces the whole subtree instead of merging it.

---

## 3. Python shim

`graphids/config/jsonnet.py::render_config(path, tla) -> dict` is the
single site that shells out to the `jsonnet` binary. Every TLA is passed
as `--tla-code <k>=<json.dumps(v)>` so ints, bools, lists, and `None`
round-trip exactly. The binary is cached via `functools.lru_cache` to
avoid repeated `shutil.which` calls.

See `docs/decisions/0010-jsonnet-binary.md` for version pins + install.

---

## 3a. Pydantic validation layer (Phase 2)

`graphids/config/schemas.py::validate_config(rendered) -> ValidatedConfig`
is the structural gate that runs **immediately after** `render_config` on
every path. Torch-free, deterministic, and fast enough to run once per
asset without cache.

### Schema tree

```
ValidatedConfig (extra="forbid")
├── seed_everything: int
├── trainer: TrainerSection    (extra="allow" — Lightning Trainer has ~50 kwargs)
├── data: ClassPathBlock       (extra="forbid"; class_path required)
├── model: ClassPathBlock      (extra="forbid"; class_path required)
├── checkpoint: CheckpointSection  (mode: Literal["min","max"])
├── early_stopping: EarlyStoppingSection  (mode: Literal["min","max"])
└── ckpt_path: str | None      (auto-resume passthrough)
```

### Model validators (migrated from `resolve._convention_errors`)

| Validator | Rule | Why it exists |
|---|---|---|
| `_no_null_list_fields` | `model.init_args.{pool_aggrs, hidden_dims, auxiliaries}` must not be null | jsonargparse rejects these at instantiation with a cryptic error |
| `_monitor_pair_consistent` | `checkpoint.monitor/mode == early_stopping.monitor/mode` | Divergent monitors = typo in the stage libsonnet (promoted from warning → error in Phase 2) |
| `_lr_monitor_requires_logger` | `LearningRateMonitor` callback needs `trainer.logger != False` | LR monitor is silently disabled without a logger |
| `_class_paths_namespaced` | `data.class_path` and `model.class_path` must start with `graphids.` or `pytorch_lightning.` | Catches relative imports and stray modules |

Stage-archetype monitor mismatches (fusion must be `val_acc/max`, every
other stage `val_loss/min`) remain a **warning** in
`orchestrate/resolve/resolver._warn_stage_monitor_mismatch` because they're
advisory, not fatal — `ValidatedConfig` already forces internal
consistency.

### Integration points

| Call site | What it does |
|---|---|
| `ConfigResolver.resolve()` | Calls `validate_config(rendered)` after `render_config`; attaches the typed view to `ResolvedConfig.validated` |
| `ConfigResolver.resolve_and_validate()` | Thin alias for `resolve()` (Phase 3 deleted the jsonargparse second pass). `dagster/assets.py` and orchestrate/validate.py use this name to signal "fully validated". |
| `train_entrypoint._instantiate_from_spec()` | Calls `validate_config` on the SLURM worker before `instantiate(...)` — belt-and-braces second pass |
| `instantiate()` | Re-validates if caller didn't pass a `ValidatedConfig` — ensures downstream code can trust `run.merged` shape |

---

## 4. Forced Callbacks + direct instantiation

Jsonnet deep-merge replaces lists atomically, same as YAML. A stage jsonnet
that sets `trainer.callbacks: [X]` drops everything else. Critical callbacks
(ModelCheckpoint, EarlyStopping, DeviceStatsMonitor, ResourceProfileCallback,
RunRecordCallback) are protected by living at top-level namespaces in the
rendered dict — `checkpoint.*`, `early_stopping.*`, etc. — and being
constructed explicitly by `core/instantiate._build_callbacks()` rather than
read from `trainer.callbacks`. Any stage-level override of `trainer.callbacks`
appends user callbacks; it cannot drop the forced set.

Defaults live in `configs/_lib/defaults.libsonnet` (baked into every stage
jsonnet). The Pydantic `CheckpointSection` / `EarlyStoppingSection` classes
in `config/schemas.py` enforce `mode: Literal["min", "max"]` and
a non-empty `monitor` string, so stage overrides that typo the monitor or
mode die at planning time.

### instantiate() responsibilities

`graphids.core.instantiate.instantiate(rendered, validated=None)` owns
every step that `GraphIDSCLI` used to own:

| Step | Old (LightningCLI) | New (Phase 3) |
|---|---|---|
| Class-path import | `jsonargparse._resolvers` | `importlib.import_module` + `getattr` |
| link_arguments | `parser.link_arguments(src, tgt)` | `_apply_link_arguments(merged, dm_cls, model_cls)` — signature-filtered |
| Forced callbacks | `parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")` | `_build_callbacks(merged, default_root_dir)` constructs the 5-callback tuple explicitly |
| Path patching | `before_instantiate_classes → patch_config_paths` | inline in `_build_callbacks` (checkpoint dirpath) and `_build_loggers` (Wandb/CSV save_dir) |
| Wandb config forward | `WandbSaveConfigCallback.save_config` | iterate `trainer.loggers`, push `rendered` dict |
| KD auxiliaries | jsonargparse Namespace wrapping of TypedDict list items | `_coerce_kd_auxiliaries` → `SimpleNamespace` |
| seed_everything | `LightningCLI(seed_everything_default=42)` | explicit `pl.seed_everything(merged["seed_everything"], workers=True)` |

---

## 5. Key Files

| File | Role | Torch? |
|---|---|---|
| `commands/train.py` | Dev-path argparse entry — `fit/test/validate/predict`, `--config`, `--tla`, `--set`, `--ckpt_path` | Lazy |
| `callbacks.py` | `ResourceProfileCallback`, `RunRecordCallback` (plain `pl.Callback` subclasses) | Yes |
| `core/instantiate.py` | `instantiate(rendered) → InstantiatedRun` — importlib class_path, signature-filtered link_arguments, forced callbacks, wandb forwarding | Yes |
| `__main__.py` | CLI dispatch: lightning commands → `commands.train.main`, others → command module dict | Lazy |
| `config/jsonnet.py` | `render_config(path, tla)` subprocess shim | No |
| `config/schemas.py` | `ValidatedConfig`, `validate_config`, `ConfigValidationError` | No |
| `config/yaml_utils.py` | `read_yaml` / `write_yaml` (snapshots + recipes) | No |
| `orchestrate/contracts/__init__.py` | `TrainingSpec` (Pydantic) — `jsonnet_path`, `jsonnet_tla`, `build_tla_dict` | No |
| `core/train_entrypoint.py` | `render_config → validate_config → snapshot → instantiate` | Yes |
| `config/contracts.py` | `TrainingRunConfig`, `KDEntry`, `expand_recipe_configs` | No |
| `config/topology.py` | Stage DAG, valid types/scales, import-time assertions against `configs/` | No |
| `config/shared.py` | `StageConfig`, `ResourceSpec` | No |
| `orchestrate/planning/planner.py` | `enumerate_assets` (StageConfig lives in `config/shared.py`) | No |
| `orchestrate/resolve/resolver.py` | `ConfigResolver` — builds TLA, renders, validates, cross-field checks via `config/schemas.py` | No |
| `orchestrate/dagster/assets.py` | `make_training_asset` | No |
| `orchestrate/dagster/component.py` | `SlurmTrainingComponent` (dagster Component) | No |

---

## 6. Architecture evaluation

### Strengths

| # | Strength | Why it matters |
|---|---|---|
| S1 | **Single composition primitive** — jsonnet replaces custom deep-merge + dotted-override + stringification plumbing with a language built for it | ~100 LOC of Python merge code deleted; no dual merge semantics to keep in sync. |
| S2 | **Torch-free config boundary** — `jsonnet.py`, `schemas.py`, `contracts/ops.py`, and the resolver never import torch; `callbacks.py` and `core/instantiate.py` lazy-imported from `commands/train.py` and `train_entrypoint.py` | Dagster workers and login nodes render and validate configs without GPU. |
| S3 | **Typed TLA round-trip** — `render_config` JSON-encodes every TLA via `--tla-code`, so ints stay ints, bools stay bools, lists stay lists | Removes the pre-migration stringification footgun (`to_override_dict` cast everything to `str`). |
| S4 | **Single convergence point** — every path (dev, pipeline, validate) ends at `instantiate(rendered, validated=...)` consuming a dict produced by `render_config` and gated by `validate_config` | No separate code paths to drift apart. |
| S5 | **Forced callbacks via explicit construction** — `_build_callbacks` assembles the 5-callback tuple from top-level sections, user callbacks from `trainer.callbacks` are appended | Prevents stage jsonnets from dropping critical callbacks while still letting them add `LearningRateMonitor` etc. |
| S6 | **Import-time config validation** — `topology.py` cross-validates the jsonnet tree + resource profiles against the declared topology at package import | Missing stage jsonnet or model libsonnet fails before any code runs. |
| S7 | **Pydantic `extra="forbid"`** — `TrainingSpec`, `TrainingRunConfig`, `KDEntry`, `ValidatedConfig` | Typos caught at construction time. |
| S8 | **Content-addressed run dirs** — `compute_identity_hash()` from `identity_keys` | Deterministic, filesystem-navigable, resumable. |
| S9 | **Typed rendered-dict gate** — `validate_config` runs Pydantic validators on the jsonnet output before any downstream consumer (Phase 2) | Structural errors, null list fields, monitor mismatches, and un-namespaced class_paths die at planning time with actionable messages instead of bubbling out of jsonargparse/torch with cryptic tracebacks. |
| S10 | **Direct instantiation via importlib** — Phase 3 replaced `GraphIDSCLI` + `jsonargparse.parse_object` with `graphids.core.instantiate.instantiate`, which imports `class_path` via `importlib` and applies signature-filtered link_arguments | Stack traces go straight to `VGAEModule.__init__` / `CANBusDataModule.__init__` instead of 15 layers of jsonargparse. KD auxiliary handling is a 3-line `SimpleNamespace` coercion, not a TypedDict+Namespace dance. |

### Known limitations

| # | Issue | Severity | Mitigation |
|---|---|---|---|
| L1 | jsonnet rendering shells out per-render (~5 ms subprocess cost) | Low | Parity harness renders ~100 configs in 500 ms; not a hot path. Swap in `_gojsonnet` bindings if needed. |
| L2 | `jsonargparse` remains only in `commands/analyze.py` (Analyzer config) — `LightningCLI` fully removed | Low | Phase 4 retools analyze configs to Jsonnet + `parser_mode="jsonnet"` while keeping jsonargparse. |
| L3 | Fusion stage absorbs `auxiliaries` + `vgae_ckpt_path` as unused TLAs because `build_tla_dict` always emits them | Low | Could filter TLAs per stage; current form is simpler. |
| L4 | No recipe schema versioning | Low | Add `version: 1` to `_RecipeEnvelope` in a later phase. |

### Comparison matrix

| Dimension | KD-GAT (post-Phase-1) | LightningCLI | Hydra | MMEngine |
|---|---|---|---|---|
| Composition | jsonnet + render shim | YAML via jsonargparse | OmegaConf interpolation | Python/YAML |
| Torch-free render | Yes | No | Yes | No |
| Type round-trip | Native (TLA JSON) | Parser-coerced | String-biased | Python objects |
| Multi-stage DAG | topology.py + dagster | No | No | No |
| Sweep support | Recipe YAML expansion | None | `--multirun` | None |
| Reproducibility | snapshot + identity hash + W&B | `SaveConfigCallback` | `outputs/` dir | `work_dir` dump |
