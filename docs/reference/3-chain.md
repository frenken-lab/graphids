Here's the current state of the 3-stage chain, with evidence for every claim.

## What dagster actually takes in

**Exactly three inputs, all read at `SlurmTrainingComponent.build_defs()` time:**

| Input             | Source                                                               | Read at     | File:line            |
| ----------------- | -------------------------------------------------------------------- | ----------- | -------------------- |
| Recipe YAML       | `KD_GAT_RECIPE` env var (defaults to `config/recipes/ablation.yaml`) | build_defs  | `component.py:86-87` |
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

Dev path (`python -m graphids fit --config ... --model.init_args.lr=0.01`) bypasses dagster entirely — that's the LightningCLI route, documented in CLAUDE.md under "Training".

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
│       ├─ merge trainer + stage + kd + resource overrides        │
│       ├─ apply_resource_overrides → ResourceSpec                │
│       ├─ _validate_cross_fields    ← num_workers≤cpus-1,        │
│       │                              curriculum epoch sync,    │
│       │                              GPU partition, RL         │
│       │                              batch_size dead-config    │
│       └─ validate_cli_chain        ← merge_yaml_chain +         │
│                                      jsonargparse.parse_object  │
│                                      catches: typos, bad keys, │
│                                      bad types, null lists,    │
│                                      callback/logger wiring    │
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
│  TrainingContract.to_override_dict(spec)                        │
│       ▼                                                         │
│  merge_yaml_chain(config_files, overrides)                      │
│       ▼                                                         │
│  parser.parse_object(merged)        ← SECOND jsonargparse       │
│       ▼                                pass (safety net;        │
│  parser.instantiate_classes(cfg)     the one that matters       │
│       ▼                                was at handoff 1)        │
│  trainer.fit(model, datamodule=data)                            │
└─────────────────────────────────────────────────────────────────┘
```

That's **three handoffs, two boundaries**. Everything between the boundaries is either pure data (`TrainingSpec` on the wire) or pure instantiation (`parser.instantiate_classes` on the SLURM side). Zero additional merges, zero additional override sources, zero string round-trips.

## Where validation catches what

| Failure mode                                                         | Caught at                                        | How                                                          |
| -------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| Recipe YAML has an unknown key (`sedes` instead of `seeds`)          | Handoff 1, step 1                                | `expand_recipe_configs` pydantic model with `extra="forbid"` |
| Invalid stage, scale, fusion_method, conv_type, loss_fn              | Handoff 1, step 1                                | `TrainingRunConfig` field validators (`contracts.py:73-113`) |
| KD alpha out of [0,1], invalid teacher_scale                         | Handoff 1, step 1                                | `KDEntry` field validators (`contracts.py:25-44`)            |
| Pipeline topology references a missing model config file             | Import time                                      | `topology.py` import-time assertions                         |
| Missing identity keys for a stage                                    | Handoff 1, step 2                                | `compute_identity_hash` raises `KeyError`                    |
| `trainer_overrides.trainer.max_epoch` (typo in trainer key)          | **Handoff 1, step 3** (via `validate_cli_chain`) | `parser.parse_object(merged)` rejects unknown dotted key     |
| Wrong type on an override value (`lr: "high"`)                       | **Handoff 1, step 3**                            | jsonargparse type coercion failure                           |
| `num_workers > cpus_per_task - 1`                                    | **Handoff 1, step 3**                            | `_validate_cross_fields` in resolve.py                       |
| `data.init_args.num_workers` in YAML > cpus                          | **Handoff 1, step 3**                            | `_validate_cross_fields` reads merged YAML                   |
| `CurriculumDataModule.max_epochs != trainer.max_epochs`              | **Handoff 1, step 3**                            | `_validate_cross_fields` catches the sync gap                |
| `gres=gpu:1` with `partition=cpu`                                    | **Handoff 1, step 3**                            | `_validate_cross_fields`                                     |
| Fusion RL (`dqn`/`bandit`) with a `batch_size` override              | **Handoff 1, step 3**                            | `_validate_cross_fields` — warns, not raises                 |
| `pool_aggrs`, `hidden_dims`, `auxiliaries` serialized as `null`      | **Handoff 1, step 3**                            | `_convention_errors` in `validate_cli_chain`                 |
| `LearningRateMonitor` with `trainer.logger=false`                    | **Handoff 1, step 3**                            | `_convention_errors`                                         |
| KD auxiliaries JSON blob is syntactically valid but semantically bad | **Handoff 1, step 3**                            | `parse_object` catches after resolve's json.dumps round-trip |
| Physically bad config that somehow slipped past step 3               | Handoff 3 (safety net)                           | `parse_object` runs again on the SLURM worker                |

**Everything fatal is caught before sbatch.** The SLURM side's `parse_object` is redundant — it runs again as a safety net, but by construction it can never fire unless someone corrupts the JSON envelope in transit. The validation desert that ADR 0009 fixed is gone.

## What the resolver "catches as early as possible" in one sentence

`ConfigResolver.resolve_and_validate()` — called from `assets._train` exactly once per materialization — runs the full override merge + cross-field validation + jsonargparse schema check + convention checks in a single pass, **before** `submit_and_wait` is invoked. After that point, the SLURM job sees exactly one JSON envelope and does nothing except deserialize and instantiate. No additional merges, no additional validators, no additional override sources.

## What the "minimum passthroughs" look like in practice

After handoff 1, the only things that cross the boundary are the fields of `TrainingSpec`:

```python
# graphids/core/contracts/ops.py (current shape)
TrainingSpec(
    stage, model_family, scale, dataset, seed, run_dir,
    config_files,                  # tuple[str, ...]  — YAML chain
    model_init_overrides,          # dict[str, Any]   — per-model tweaks
    upstream_ckpt_paths,           # dict[str, str]   — for KD/staged handoff
    upstream_model_families,       # dict[str, str]
    runtime_overrides,             # dict[str, str]   — everything from recipe
)
```

`runtime_overrides` is a **single flat dict of dotted keys → string values**, e.g.:

```python
{
    "trainer.max_epochs": "50",
    "data.init_args.max_epochs": "50",            # from stage_overrides
    "model.init_args.auxiliaries": '[{"type": "kd", ...}]',  # KD blob
}
```

Every override source (trainer, stage, KD, per-stage) gets flattened into this dict at handoff 1, inside `ConfigResolver.resolve()`:

```python
# resolve.py:121-139 (verified in this session)
runtime_overrides: dict[str, Any] = {}
if cfg.trainer_overrides:
    runtime_overrides.update(cfg.trainer_overrides)
if cfg.stage_overrides:
    runtime_overrides.update(cfg.stage_overrides)
if cfg.kd_overrides:
    runtime_overrides["model.init_args.auxiliaries"] = json.dumps([cfg.kd_overrides])
```

SLURM side then calls `TrainingContract.to_override_dict(spec)` (adds the non-overridable identity fields — `dataset`, `seed_everything`, `trainer.default_root_dir`, upstream ckpt flags) and hands the result to `merge_yaml_chain` + `parser.parse_object`. No additional merge layers, no additional CLI construction, no additional validation gates that could reject what handoff 1 already blessed.

## Is there any remaining passthrough that could be collapsed?

I'd flag two items — neither critical, both in the **catch-early** direction:

**1. `TrainingContract.to_override_dict` is called twice in the lifetime of a run** — once implicitly by `ConfigResolver.validate_cli_chain` (which rebuilds the merged dict to validate), and once on the SLURM side at `from-spec --phase train`. If we persisted the merged dict directly in the envelope instead of reconstructing it on the SLURM side, the SLURM pass becomes: `json.load → parser.parse_object → instantiate_classes`. ~20 lines gone from `train_entrypoint`, one less rebuild step. Tradeoff: envelopes grow because they carry the full merged YAML. Small regression in envelope size vs small improvement in fidelity — debatable.

**2. `runtime_overrides` stringifies everything** — `_s = lambda v: str(v).lower() if isinstance(v, bool) else str(v)` in `to_override_dict`. This is because `runtime_overrides` is `dict[str, str]` per the `TrainingSpec` type. It forces `parser.parse_object` to coerce back to the right type on the other side. Not a correctness issue (jsonargparse handles the coercion), but the type stringification is unnecessary now that we control both endpoints of handoff 2. Could be `dict[str, Any]` throughout. Savings: maybe 5 LOC in contracts + some clarity.

Both are cleanups, not correctness fixes. The chain at 3 stages is already the right shape for the ADR's goals.

**In short**: dagster takes in exactly one env var (`KD_GAT_RECIPE`) pointing at one YAML file. All override flavors flow through that file, get flattened into `runtime_overrides` inside `ConfigResolver`, get jsonargparse-validated at planning time (pre-sbatch), then cross one JSON boundary to a SLURM worker that does no merging of its own. The "CLI override" story is: there is no runtime CLI for the pipeline path — all variation lives in the recipe, and the recipe is the single input that ConfigResolver turns into a fully-validated asset plan before anything hits the queue.
