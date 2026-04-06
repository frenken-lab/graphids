Here's the current state of the 3-stage chain, with evidence for every claim.

## What dagster actually takes in

**Exactly three inputs, all read at `SlurmTrainingComponent.build_defs()` time:**

| Input             | Source                                                               | Read at     | File:line            |
| ----------------- | -------------------------------------------------------------------- | ----------- | -------------------- |
| Recipe Jsonnet    | `KD_GAT_RECIPE` env var (defaults to `configs/recipes/ablation.jsonnet`) | build_defs  | `component.py:86-87` |
| Pipeline topology | `PIPELINE_YAML` (static, from `topology.py`)                         | import time | `component.py:91`    |
| Dataset catalog   | `dataset_names()` → per-dataset YAMLs                                | build_defs  | `component.py:94`    |

**There is no dagster-level CLI override mechanism for pipeline runs.** Dagster's configurable resources (e.g. `SlurmTrainingResource.dry_run`) are set at Component construction (`definitions.py:32` reads `KD_GAT_DRY_RUN`), not per-run. All per-run variation goes through the **recipe YAML**, which carries four blocks of override data:

```yaml
# smoke_test.yaml (verified example)
seeds: [99]
selection:
  datasets: [hcrl_sa]
  model_families: [vgae, gat, fusion]
  scales: [small]
  stages: { vgae: [autoencoder], gat: [normal, curriculum], fusion: [fusion] }
  fusion_methods: [bandit, dqn, mlp, weighted_avg]
trainer_overrides: # → runtime_overrides, all stages
  trainer.max_epochs: 50
stage_overrides: # → runtime_overrides, scoped to one stage
  curriculum:
    data.init_args.max_epochs: "50"
resource_overrides: # → ResourceSpec.apply_resource_overrides
  time: "1:00:00"
  partition: gpudebug
```

The only "CLI override" in the dagster path is environment-level:

- `KD_GAT_RECIPE=/path/to/recipe.yaml` — swap recipes
- `KD_GAT_DRY_RUN=1` — skip sbatch, return path-only
- `KD_GAT_LAKE_ROOT=/fs/ess/...` — change output root
- `USER` — dev vs production namespace

Dev path (`python -m graphids fit --config configs/stages/autoencoder.jsonnet --set model.init_args.lr=0.01`) bypasses dagster entirely — that's the `commands/train.py` argparse route, documented in CLAUDE.md under "Training".

## The 3 handoffs (per ADR 0009)

```
┌─────────────────────────────────────────────────────────────────┐
│ HANDOFF 1: Plan  (dagster worker, CPU, pre-SLURM)              │
│                                                                 │
│  KD_GAT_RECIPE ─► read_yaml                                    │
│       ▼                                                         │
│  expand_recipe_configs()         ← Pydantic: recipe shape       │
│       ▼                                                         │
│  enumerate_assets(pipeline, recipe)  ← TrainingRunConfig        │
│       │                                validates identity       │
│       ▼                                fields (stages, scale,   │
│  list[StageConfig]                     conv_type, KDEntry)      │
│       ▼                                                         │
│  ConfigResolver.resolve_and_validate(cfg, dataset, seed)        │
│       ├─ build jsonnet_tla (trainer+stage+kd+upstream ckpts)   │
│       ├─ apply_resource_overrides → ResourceSpec                │
│       ├─ render_config(jsonnet_path, jsonnet_tla)              │
│       ├─ validate_config(rendered)  ← Pydantic ValidatedConfig  │
│       │                              catches: extra top-level  │
│       │                              keys, null list fields,   │
│       │                              monitor mismatch,         │
│       │                              un-namespaced class_path, │
│       │                              LR monitor + logger=false │
│       └─ validate_stage_config     ← num_workers≤cpus-1,        │
│                                      curriculum epoch sync,    │
│                                      GPU partition, RL         │
│                                      batch_size dead-config    │
│       ▼                                                         │
│  ResolvedConfig(TrainingSpec, ResourceSpec, PathContext, audit) │
└──────────┬──────────────────────────────────────────────────────┘
           │
           │  ── SERIALIZATION BOUNDARY (JSON envelope) ──
           │
┌──────────▼──────────────────────────────────────────────────────┐
│ HANDOFF 2: Submit  (dagster worker, still CPU)                 │
│                                                                 │
│  TrainingContract.to_envelope(training_spec)                    │
│       ▼                                                         │
│  write_training_spec → /fs/.../specs/<job>_<uuid>.json          │
│       ▼                                                         │
│  generate_script(resources, spec_file, ...)                     │
│       ▼                                                         │
│  sbatch ─► SLURM queue                                          │
└──────────┬──────────────────────────────────────────────────────┘
           │
           │  ── RUN BOUNDARY (sbatch → GPU node) ──
           │
┌──────────▼──────────────────────────────────────────────────────┐
│ HANDOFF 3: Run  (SLURM worker, GPU)                            │
│                                                                 │
│  python -m graphids from-spec --phase train --spec-file X.json  │
│       ▼                                                         │
│  TrainingContract.from_envelope(X)  → TrainingSpec              │
│       ▼                                                         │
│  render_config(spec.jsonnet_path, spec.jsonnet_tla)             │
│       ▼                                                         │
│  validate_config(merged)            ← belt-and-braces           │
│       ▼                                Pydantic re-validation   │
│  snapshot_config(merged, run_dir)   ← config_snapshot.yaml      │
│       ▼                                                         │
│  instantiate(merged, validated=...) ← importlib class_paths,    │
│       ▼                                forced callbacks,        │
│  trainer.fit(model, datamodule=data)  signature-filtered links  │
└─────────────────────────────────────────────────────────────────┘
```

That's **three handoffs, two boundaries**. Everything between the boundaries is either pure data (`TrainingSpec` on the wire) or pure instantiation (`graphids.core.instantiate.instantiate` on the SLURM side). Zero additional merges, zero additional override sources, zero string round-trips.

## Where validation catches what

| Failure mode                                                         | Caught at                                        | How                                                          |
| -------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| Recipe YAML has an unknown key (`sedes` instead of `seeds`)          | Handoff 1, step 1                                | `expand_recipe_configs` pydantic model with `extra="forbid"` |
| Invalid stage, scale, fusion_method, conv_type, loss_fn              | Handoff 1, step 1                                | `TrainingRunConfig` field validators (`contracts.py:73-113`) |
| KD alpha out of [0,1], invalid teacher_scale                         | Handoff 1, step 1                                | `KDEntry` field validators (`contracts.py:25-44`)            |
| Pipeline topology references a missing model config file             | Import time                                      | `topology.py` import-time assertions                         |
| Missing identity keys for a stage                                    | Handoff 1, step 2                                | `compute_identity_hash` raises `KeyError`                    |
| `trainer_overrides.trainer.max_epoch` (typo in trainer key)          | Handoff 3 (instantiation)                        | stage libsonnet merge — typo is silently accepted as a new dotted key; caught when the Trainer constructor rejects the arg. Pre-Phase-3 `validate_cli_chain` used to catch this at handoff 1; consider re-adding stage-libsonnet override validation if this becomes a pain point. |
| Wrong type on an override value (`lr: "high"`)                       | Handoff 3 (instantiation)                        | `VGAEModule.__init__` rejects non-float `lr` via Python's type system |
| `num_workers > cpus_per_task - 1`                                    | **Handoff 1, step 3**                            | `validate_stage_config` (`config/schemas.py`)                |
| `data.init_args.num_workers` in rendered config > cpus               | **Handoff 1, step 3**                            | `validate_stage_config` reads rendered dict                  |
| `CurriculumDataModule.max_epochs != trainer.max_epochs`              | **Handoff 1, step 3**                            | `validate_stage_config` catches the sync gap                 |
| `gres=gpu:1` with `partition=cpu`                                    | **Handoff 1, step 3**                            | `validate_stage_config`                                      |
| Fusion RL (`dqn`/`bandit`) with a `batch_size` override              | **Handoff 1, step 3**                            | `validate_stage_config` reads `spec.jsonnet_tla`             |
| `pool_aggrs`, `hidden_dims`, `auxiliaries` serialized as `null`      | **Handoff 1, step 3**                            | `ValidatedConfig._no_null_list_fields` model validator       |
| `LearningRateMonitor` with `trainer.logger=false`                    | **Handoff 1, step 3**                            | `ValidatedConfig._lr_monitor_requires_logger` model validator |
| `checkpoint` + `early_stopping` track different monitors/modes       | **Handoff 1, step 3**                            | `ValidatedConfig._monitor_pair_consistent` model validator   |
| `data.class_path` / `model.class_path` not under a known namespace   | **Handoff 1, step 3**                            | `ValidatedConfig._class_paths_namespaced` model validator    |
| Extra top-level key in rendered dict (typo at stage libsonnet level) | **Handoff 1, step 3**                            | `ValidatedConfig` has `extra="forbid"` on root               |
| KD auxiliaries malformed (bad keys on a ``KDEntry``)                 | **Handoff 1, step 1**                            | `KDEntry` Pydantic validator (pre-TLA)                       |
| Physically bad config that slipped past step 3                       | Handoff 3 (safety net)                           | `validate_config` runs again on the SLURM worker             |

**Structural failures are caught before sbatch.** The SLURM side's `validate_config` is redundant — it runs again as a safety net, but by construction it can only fire if the JSON envelope was corrupted in transit. The validation desert that ADR 0009 fixed is gone. (Phase 3 narrowed the set of failures caught at handoff 1: dotted-key typos that the pre-Phase-3 jsonargparse pass used to catch now fail at instantiation time instead.)

## What the resolver "catches as early as possible" in one sentence

`ConfigResolver.resolve_and_validate()` — called from `assets._train` exactly once per materialization — runs the full override merge + jsonnet render + Pydantic `ValidatedConfig` gate + cross-field validation in a single pass, **before** `submit_and_wait` is invoked. After that point, the SLURM job sees exactly one JSON envelope and does nothing except deserialize and instantiate. No additional merges, no additional validators, no additional override sources.

## What the "minimum passthroughs" look like in practice

After handoff 1, the only things that cross the boundary are the fields of `TrainingSpec`:

```python
# graphids/core/contracts/models.py (post Phase 1)
TrainingSpec(
    stage, model_family, scale, dataset, seed, run_dir,
    jsonnet_path,                  # str              — configs/stages/<stage>.jsonnet
    jsonnet_tla,                   # dict[str, Any]   — typed TLA dict
    model_init_overrides,          # dict[str, Any]   — identity-derived per-model tweaks
    upstream_ckpt_paths,           # dict[str, str]   — for KD/staged handoff
    upstream_model_families,       # dict[str, str]
)
```

`jsonnet_tla` is a **typed dict matching the stage function's TLA signature** — ints stay ints, bools stay bools, recipe overrides stay as sub-dicts. Example for an autoencoder stage:

```python
{
    "dataset": "hcrl_ch",
    "seed": 42,
    "run_dir": "/fs/.../autoencoder_abc123/seed_42",
    "scale": "small",
    "conv_type": "gatv2",
    "variational": True,
    "auxiliaries": [],                               # empty = no KD
    "vgae_ckpt_path": None,
    "trainer_overrides": {"trainer.max_epochs": "50"},
    "stage_overrides": {},
}
```

Every override source (trainer, stage, KD, upstream ckpts) is packed into this dict at handoff 1, inside `ConfigResolver.resolve()` via `TrainingContract.build_tla_dict()`:

```python
# core/contracts/ops.py
tla = TrainingContract.build_tla_dict(
    cfg,
    dataset=dataset, seed=seed, run_dir=run_dir,
    upstream_ckpts=upstream_ckpts,
    upstream_model_families=cfg.upstream_model_families,
    kd_overrides=cfg.kd_overrides or None,
    trainer_overrides=cfg.trainer_overrides or None,
    stage_overrides=cfg.stage_overrides or None,
)
```

SLURM side then calls `render_config(spec.jsonnet_path, spec.jsonnet_tla)`, re-runs `validate_config` as a belt-and-braces gate, snapshots `config_snapshot.yaml`, and hands the rendered dict to `graphids.core.instantiate.instantiate`. No additional merge layers, no additional CLI construction, no stringification round-trip, no additional validation gates that could reject what handoff 1 already blessed.

**In short**: dagster takes in exactly one env var (`KD_GAT_RECIPE`) pointing at one YAML file. All override flavors flow through that file, get packed into `jsonnet_tla` inside `ConfigResolver`, get jsonnet-rendered + jsonargparse-validated at planning time (pre-sbatch), then cross one JSON boundary to a SLURM worker that does no merging of its own. The "CLI override" story is: there is no runtime CLI for the pipeline path — all variation lives in the recipe, and the recipe is the single input that ConfigResolver turns into a fully-validated asset plan before anything hits the queue.
