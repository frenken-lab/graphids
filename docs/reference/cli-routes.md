# CLI + Config Architecture Reference

> Generated: 2026-04-01 | Source of truth for CLI routes and config resolution

---

## CLI Routes

There are **3 routes** that end in training, plus operational commands:

### Route A: Dev CLI (interactive)

```
python -m graphids fit --config stage.yaml --config model/base.yaml --model.init_args.lr=0.01
  -> __main__.py
  -> _run_lightning("fit", argv)
  -> torch.mp.set_start_method("spawn")
  -> cli.run_lightning(["fit", "--config", ...])
  -> GraphIDSCLI(LightningCLI)(**CLI_KWARGS, args=args)
  -> jsonargparse merges + validates + instantiates
  -> trainer.fit(model, datamodule)
```

User provides `--config` flags and `--key=value` overrides directly.

### Route B: Pipeline (dagster -> SLURM -> LightningCLI)

```
dg launch --assets '*'
  -> dagster -> SlurmTrainingComponent.build_defs()
    -> expand_recipe_configs(recipe)           # combinatorial expansion
    -> enumerate_assets(PIPELINE_YAML, recipe) # -> list[StageConfig]
    -> make_training_asset(cfg, ...)
      -> ConfigResolver.resolve(cfg, ...)      # cross-field validation + audit
      -> SlurmTrainingResource.submit_and_wait(spec, resources)
        -> sbatch -> SLURM job:

          python -m graphids train-from-spec --spec-file /tmp/spec.json
            -> __main__.py -> _run_module(...)
            -> train_from_spec.main(argv)
            -> run_training_from_payload(payload)
            -> run_training_from_spec(spec)
              -> resolve_configs(...)          # snapshot only (reproducibility)
              -> write_yaml(snapshot)
              -> torch.mp.set_start_method("spawn")
              -> run_lightning(_build_cli_args(spec))    <-- same endpoint as Route A
              -> GraphIDSCLI(LightningCLI)
              -> trainer.fit(model, datamodule)
```

Dagster builds `TrainingSpec`, serializes to JSON, submits via SLURM. The SLURM job
deserializes, builds CLI args, and calls `run_lightning()` -- same endpoint as Route A.

### Route C: Validation (parse-only, no training)

```
python -m graphids.orchestrate validate
  -> validate_recipe(argv)
  -> enumerate_assets(PIPELINE_YAML, recipe)
  -> For each StageConfig:
    -> resolve_configs(config_files, overrides) -> write snapshot
    -> GraphIDSCLI(run=False, args=["--config", snapshot])   <-- parse, don't run
    -> check: null list fields, logger/callback compat, monitor conventions
```

### Route D: Operational commands (no LightningCLI)

```
python -m graphids {analyze|landscape|profile|rebuild-caches|stage-data|...}
  -> __main__.py -> _run_module(module_name, argv)
  -> each command has its own argparse + logic, no LightningCLI
```

**Key invariant:** Routes A and B converge at `run_lightning()` -> `GraphIDSCLI(LightningCLI)`.
One instantiation path.

---

## Config Resolution

### Merge order (left-to-right, last wins)

```
Layer 0 (lowest priority):
  defaults/trainer.yaml              <-- CLI_KWARGS.default_config_files
                                        (trainer settings, forced callback defaults)

Layer 1-N (--config chain, L->R):
  stages/{stage}.yaml                <-- model class_path, data class_path, stage-specific
  models/{family}/base.yaml          <-- shared architecture defaults for model family
  models/{family}/scales/{scale}.yaml <-- hidden_dims, latent_dim, num_layers
  [fusion/base.yaml]                 <-- fusion-only: shared fusion defaults
  [fusion/methods/{method}.yaml]     <-- fusion-only: method overlay (bandit/dqn/mlp)
  [models/{family}/kd.yaml]          <-- KD-only: auxiliary config overlay

Layer N+1 (highest priority):
  CLI --key=value overrides          <-- dataset, seed, run_dir, model_init_overrides,
                                        upstream ckpt paths, runtime overrides
```

jsonargparse merges these. Later `--config` files override earlier ones at the dict level
(deep merge, lists replaced atomically). CLI `--key=value` overrides win over everything.

### Who builds each layer

| Layer | Dev path (Route A) | Pipeline path (Route B) |
|---|---|---|
| Layer 0 | `CLI_KWARGS.default_config_files` | Same -- `run_lightning` uses same `CLI_KWARGS` |
| Layer 1-N | User types `--config` flags | `TrainingContract.resolve_config_files(stage, scale, ...)` |
| CLI overrides | User types `--key=val` | `TrainingContract.to_override_dict(spec)` -> `_build_cli_args` |

### resolve_config_files output by stage

```python
# autoencoder/normal/curriculum:
("stages/autoencoder.yaml", "models/vgae/base.yaml", "models/vgae/scales/small.yaml")

# fusion:
("stages/fusion.yaml", "fusion/base.yaml", "fusion/methods/bandit.yaml")

# with KD overlay:
("stages/normal.yaml", "models/gat/base.yaml", "models/gat/scales/small.yaml", "models/gat/kd.yaml")
```

### to_override_dict output (becomes --key=value CLI args)

```python
{
    "data.init_args.dataset": "hcrl_ch",
    "seed_everything": "42",
    "trainer.default_root_dir": "/path/to/run_dir",
    "model.init_args.conv_type": "gatv2",      # from model_init_overrides
    "data.init_args.vgae_ckpt_path": "/path",   # from upstream_ckpt_paths
    "trainer.max_epochs": "5",                   # from runtime_overrides
}
```

---

## Where merging happens (and why)

There are **3 merge sites** but only **1 is for instantiation**:

| Site | Code | Purpose | Feeds instantiation? |
|---|---|---|---|
| **jsonargparse** | `GraphIDSCLI(args=...)` | Merge + type validate + instantiate | **Yes -- the only one** |
| **resolve_configs()** | `train_entrypoint.py:33` | Write `config_snapshot.yaml` | No -- reproducibility artifact |
| **ConfigResolver._merge_yaml_chain()** | `resolve.py:129` | Cross-field validation | No -- dagster-side checks only |

The last two are read-only uses of the merged state. If they disagree with jsonargparse's
merge (they shouldn't -- same files, same order), jsonargparse wins because it's the one
that instantiates.

### Naive merge vs jsonargparse merge

`resolve_configs()` and `ConfigResolver._merge_yaml_chain()` both use
`yaml_utils.merge_yaml_chain()` -- a naive `deep_merge` (recursive dict merge, lists
replaced atomically). This is NOT jsonargparse's type-aware merge. It exists because
jsonargparse requires torch imports to do a full type-aware merge, and these sites run
in torch-free contexts (login node, dagster workers).

The tradeoff: login-node/dagster-worker safety vs merge fidelity. This hasn't diverged
in practice because the config tree is shallow and list replacement is mitigated by forced
callbacks in separate namespaces (`add_lightning_class_args`).

---

## Forced Callbacks (list replacement protection)

YAML deep merge replaces lists atomically. A stage YAML with `trainer.callbacks: [X]`
drops everything from defaults. Critical callbacks (ModelCheckpoint, EarlyStopping) are
protected by registering them as **separate parser namespaces** via
`add_lightning_class_args` in `_lightning.py`. They live at `checkpoint.*` and
`early_stopping.*`, not inside `trainer.callbacks`, so list replacement cannot affect them.

Defaults are defined once in `cli.py` (`CHECKPOINT_DEFAULTS`, `EARLY_STOPPING_DEFAULTS`)
and consumed by `_lightning.py`. The same values also appear in `defaults/trainer.yaml`
for the naive merge path (snapshot, validation). These must stay in sync.

---

## Key Files

| File | Role |
|---|---|
| `cli.py` | Torch-free entry point. `resolve_configs()`, `run_lightning()`, shared wiring constants |
| `_lightning.py` | `GraphIDSCLI(LightningCLI)`, `CLI_KWARGS`, `WandbSaveConfigCallback` |
| `__main__.py` | CLI dispatch: lightning commands -> `_run_lightning`, others -> `_run_module` |
| `core/train_entrypoint.py` | Pipeline entry: `_build_cli_args(spec)` -> `run_lightning()` |
| `core/contracts/ops.py` | `TrainingContract`: `resolve_config_files`, `to_override_dict`, envelope serde |
| `core/contracts/models.py` | `TrainingSpec` (Pydantic, `extra="forbid"`) |
| `config/contracts.py` | `TrainingRunConfig` (recipe schema), `KDEntry`, `expand_recipe_configs` |
| `config/topology.py` | Stage DAG, valid types/scales/methods, import-time assertions |
| `config/paths.py` | `PathContext` (frozen Pydantic), `compute_identity_hash` |
| `config/yaml_utils.py` | `deep_merge`, `apply_dotted_overrides`, `merge_yaml_chain`, `write_yaml` |
| `orchestrate/resolve.py` | `ConfigResolver` (cross-field validation + audit trail) |
| `orchestrate/planning.py` | `StageConfig`, `enumerate_assets` (recipe -> asset list) |
| `orchestrate/assets.py` | `make_training_asset`, `make_analysis_asset` (dagster @asset factories) |
| `orchestrate/component.py` | `SlurmTrainingComponent` (dagster Component, builds Definitions) |
| `orchestrate/validate.py` | `validate_recipe` (parse-only validation of all config chains) |
| `slurm/resources.py` | `ResourceSpec`, `get_resources`, `scale_resources` |
| `slurm/slurm.py` | `SubprocessSlurmJobClient` (sbatch + sacct polling) |
| `config/defaults/trainer.yaml` | Shared trainer defaults + forced callback values |

---

## Config Directory Structure

```
graphids/config/
  defaults/
    trainer.yaml             # trainer settings + checkpoint/early_stopping defaults
    global.yaml              # global defaults (constants)
    io.yaml                  # I/O defaults
  stages/
    autoencoder.yaml         # VGAEModule class_path + init_args + data config
    normal.yaml              # GATModule (no curriculum)
    curriculum.yaml          # GATModule + CurriculumDataModule
    temporal.yaml            # TemporalLightningModule + TemporalDataModule
    fusion.yaml              # single fusion stage YAML (all methods)
    analyze_vgae.yaml        # Analyzer config: VGAE embeddings + landscape
    analyze_gat.yaml         # Analyzer config: GAT embeddings + attention + CKA
    analyze_fusion.yaml      # Analyzer config: fusion policy
  models/
    vgae/
      base.yaml              # shared VGAE architecture defaults
      scales/{small,large}.yaml
    gat/
      base.yaml              # shared GAT architecture defaults
      scales/{small,large}.yaml
    dgi/
      base.yaml
      scales/{small,large}.yaml
    temporal/
      base.yaml
      scales/{small,large}.yaml
  fusion/
    base.yaml                # shared fusion defaults
    methods/{bandit,dqn,mlp,weighted_avg}.yaml
    scales/{small,large}.yaml
  datasets/
    {hcrl_ch,hcrl_sa,set_01,...set_04}.yaml
  matrix/
    axes.yaml                # valid model types, scales, fusion methods
    allowed_combinations.yaml
  resources/
    clusters.yaml            # cluster-specific settings
    submit_profiles.yaml     # submit.sh profile mappings
    profiles/{vgae,gat,dgi,temporal,fusion}.yaml
  recipes/
    ablation.yaml            # 17-config ablation matrix
    smoke_test.yaml          # quick validation recipe
  schema/                    # JSON-schema-style validation
```
