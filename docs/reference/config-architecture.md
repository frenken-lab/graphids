# Config Architecture

> CLI routes, config resolution, and architecture evaluation.
> Consolidates: cli-routes.md, config-evaluation.md

---

## 1. CLI Routes

Three routes end in training, plus operational commands:

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

### Route B: Pipeline (dagster -> SLURM -> LightningCLI)

```
dg launch --assets '*'
  -> dagster -> SlurmTrainingComponent.build_defs()
    -> expand_recipe_configs(recipe)
    -> enumerate_assets(PIPELINE_YAML, recipe) -> list[StageConfig]
    -> make_training_asset(cfg, ...)
      -> ConfigResolver.resolve(cfg, ...)
      -> SlurmTrainingResource.submit_and_wait(spec, resources)
        -> sbatch -> SLURM job:

          python -m graphids from-spec --phase train --spec-file /tmp/spec.json
            -> from_spec.main(argv)
            -> run_training_from_spec(spec)
              -> resolve_configs(...)          # snapshot (reproducibility)
              -> run_lightning(_build_cli_args(spec))    <-- same as Route A
              -> GraphIDSCLI(LightningCLI)
              -> trainer.fit(model, datamodule)
```

### Route C: Validation (parse-only)

```
python -m graphids.orchestrate validate
  -> validate_recipe(argv)
  -> enumerate_assets(PIPELINE_YAML, recipe)
  -> For each StageConfig:
    -> resolve_configs(...) -> write snapshot
    -> GraphIDSCLI(run=False, args=["--config", snapshot])   <-- parse, don't run
    -> check: null list fields, logger/callback compat, monitor conventions
```

### Route D: Operational commands (no LightningCLI)

```
python -m graphids {analyze|landscape|profile|rebuild-caches|stage-data|...}
  -> __main__.py -> _run_module(module_name, argv)
  -> each command has its own argparse + logic
```

**Key invariant:** Routes A and B converge at `run_lightning()` -> `GraphIDSCLI(LightningCLI)`. One instantiation path.

---

## 2. Config Resolution

### Merge order (left-to-right, last wins)

```
Layer 0 (lowest priority):
  defaults/trainer.yaml              <-- CLI_KWARGS.default_config_files

Layer 1-N (--config chain, L->R):
  stages/{stage}.yaml                <-- model class_path, data class_path
  models/{family}/base.yaml          <-- shared architecture defaults
  models/{family}/scales/{scale}.yaml <-- hidden_dims, latent_dim, num_layers
  [fusion/base.yaml]                 <-- fusion-only
  [fusion/methods/{method}.yaml]     <-- fusion method overlay
  [models/{family}/kd.yaml]          <-- KD-only: auxiliary config overlay

Layer N+1 (highest priority):
  CLI --key=value overrides          <-- dataset, seed, run_dir, model overrides
```

### Who builds each layer

| Layer | Dev path (Route A) | Pipeline path (Route B) |
|---|---|---|
| Layer 0 | `CLI_KWARGS.default_config_files` | Same |
| Layer 1-N | User types `--config` flags | `TrainingContract.resolve_config_files(stage, scale, ...)` |
| CLI overrides | User types `--key=val` | `TrainingContract.to_override_dict(spec)` -> `_build_cli_args` |

### Three merge sites (only one instantiates)

| Site | Code | Purpose | Authoritative? |
|---|---|---|---|
| **jsonargparse** | `GraphIDSCLI(args=...)` | Merge + type validate + instantiate | **Yes — the only one** |
| **resolve_configs()** | `train_entrypoint.py` | Write `config_snapshot.yaml` | No — reproducibility |
| **ConfigResolver** | `resolve.py` | Cross-field validation | No — dagster-side |

The last two use `yaml_utils.merge_yaml_chain()` — a naive `deep_merge` (recursive dict merge, lists replaced atomically). This is NOT jsonargparse's type-aware merge. It exists because jsonargparse requires torch imports, and these sites run in torch-free contexts (login node, dagster workers).

---

## 3. Forced Callbacks

YAML deep merge replaces lists atomically. A stage YAML with `trainer.callbacks: [X]` drops everything from defaults. Critical callbacks (ModelCheckpoint, EarlyStopping) are protected via `add_lightning_class_args` in `_lightning.py` — they live at `checkpoint.*` and `early_stopping.*`, outside `trainer.callbacks`.

Defaults defined in `cli.py` (`CHECKPOINT_DEFAULTS`, `EARLY_STOPPING_DEFAULTS`) and mirrored in `defaults/trainer.yaml`. These must stay in sync (see weakness N3 below).

---

## 4. Key Files

| File | Role | Torch? |
|---|---|---|
| `cli.py` | Shared constants, `resolve_configs()`, `run_lightning()` | No |
| `_lightning.py` | `GraphIDSCLI(LightningCLI)`, `CLI_KWARGS`, `WandbSaveConfigCallback` | Yes |
| `__main__.py` | CLI dispatch: lightning → `_run_lightning`, others → `_run_module` | Lazy |
| `core/train_entrypoint.py` | Pipeline entry: `_build_cli_args(spec)` → `run_lightning()` | Yes |
| `core/contracts/ops.py` | `TrainingContract`: config files, overrides, serde | No |
| `core/contracts/models.py` | `TrainingSpec` (Pydantic, `extra="forbid"`) | No |
| `config/contracts.py` | `TrainingRunConfig`, `KDEntry`, `expand_recipe_configs` | No |
| `config/topology.py` | Stage DAG, valid types/scales, import-time assertions | No |
| `config/paths.py` | `PathContext`, `compute_identity_hash` | No |
| `config/yaml_utils.py` | `deep_merge`, `apply_dotted_overrides`, `merge_yaml_chain` | No |
| `orchestrate/resolve.py` | `ConfigResolver` (cross-field validation + audit trail) | No |
| `orchestrate/planning.py` | `StageConfig`, `enumerate_assets` | No |
| `orchestrate/assets.py` | `make_training_asset`, `make_analysis_asset` | No |
| `orchestrate/component.py` | `SlurmTrainingComponent` (dagster Component) | No |
| `orchestrate/validate.py` | Config chain validation | Yes |

---

## 5. Config Directory Structure

```
graphids/config/
  defaults/
    trainer.yaml             # trainer settings + checkpoint/early_stopping defaults
    global.yaml              # global defaults
    io.yaml                  # I/O defaults
  stages/
    autoencoder.yaml, normal.yaml, curriculum.yaml, temporal.yaml
    fusion.yaml, analyze_vgae.yaml, analyze_gat.yaml, analyze_fusion.yaml
  models/
    vgae/   base.yaml, scales/{small,large}.yaml
    gat/    base.yaml, scales/{small,large}.yaml
    dgi/    base.yaml, scales/{small,large}.yaml
    temporal/ base.yaml, scales/{small,large}.yaml
  fusion/
    base.yaml, methods/{bandit,dqn,mlp,weighted_avg}.yaml, scales/{small,large}.yaml
  datasets/
    {hcrl_ch,hcrl_sa,set_01,...set_04}.yaml
  matrix/
    axes.yaml                # valid model types, scales, fusion methods
  resources/
    clusters.yaml, submit_profiles.yaml
    profiles/{vgae,gat,dgi,temporal,fusion}.yaml
  recipes/
    ablation.yaml, smoke_test.yaml
```

---

## 6. Architecture Evaluation

> Independent MLOps evaluation against Hydra, MMEngine, and vanilla LightningCLI. Date: 2026-04-01.

### Strengths

| # | Strength | Why it matters |
|---|---|---|
| S1 | **Torch-free config boundary** — `cli.py` never imports torch; `_lightning.py` lazy-imported at call time | Dagster workers and login nodes can resolve configs without GPU. Neither Hydra nor MMEngine address this. |
| S2 | **Single convergence point** — all paths converge at `GraphIDSCLI(**CLI_KWARGS, args=args)` | Avoids the separate-code-path divergence common in Hydra systems (`@hydra.main` vs `compose` API). |
| S3 | **Forced callbacks via parser namespaces** — `add_lightning_class_args(ModelCheckpoint, "checkpoint")` | Prevents YAML list replacement from dropping critical callbacks. Not in LightningCLI docs. |
| S4 | **Import-time config validation** — `topology.py` cross-validates model/scale/resource configs at import | Catches missing configs earlier than Hydra (runtime) or MMEngine (runtime). |
| S5 | **Pydantic `extra="forbid"`** — `TrainingSpec`, `TrainingRunConfig`, `KDEntry` | Typos caught at construction time. |
| S6 | **Content-addressed run dirs** — `compute_identity_hash()` from `identity_keys` | Deterministic, filesystem-navigable, resumable. Better than MLflow/W&B opaque IDs or Hydra timestamp dirs. |

### Weaknesses

**Critical:**

| # | Issue | Severity | Mitigation |
|---|---|---|---|
| C1 | **Dual merge semantics** — naive `deep_merge` vs jsonargparse type-aware merge could diverge | High (low probability) | Add CI test: `assert naive_merge(chain) == jsonargparse_parse(chain)` for all recipes |
| C2 | **Non-atomic config snapshot write** — `path.write_text()` not atomic on NFS | Medium | Use `fsync` + temp-file-then-rename (5-line fix) |

**Near-term:**

| # | Issue | Severity |
|---|---|---|
| N1 | Recipe expansion growing its own type system (`_KDSpec`, `_SweepSpec` duplicate `TrainingRunConfig` fields) | Medium |
| N2 | Validation requires full Lightning import + parser instantiation | Medium |
| N3 | Checkpoint/EarlyStopping defaults duplicated in `cli.py` and `trainer.yaml` | Low |
| N4 | No config diff/change tracking between runs | Low |

**Long-term:**

| # | Issue | Severity |
|---|---|---|
| L1 | jsonargparse coupling (~2.5k stars, single maintainer) | Medium |
| L2 | No recipe schema versioning | Low |
| L3 | No dry-run mode for pipeline execution | Low |

### Recommendations (prioritized)

1. **Reconcile dual merge (C1):** CI test asserting merge equivalence for all recipe configs
2. **Atomic config writes (C2):** `fsync` + temp-file-then-rename in `write_yaml()`
3. **Eliminate callback duplication (N3):** Read YAML values in `cli.py` instead of hardcoding
4. **Recipe versioning (L2):** Add `version: 1` to `_RecipeEnvelope`
5. **Dry-run mode (L3):** Print execution plan from `enumerate_assets()` without submitting

**Not recommended:** Hydra migration (jsonargparse/LightningCLI integration is load-bearing), MMEngine (monolithic design), OmegaConf layer (would create a third merge implementation).

### Comparison Matrix

| Dimension | KD-GAT | LightningCLI | Hydra | MMEngine | Dagster Config |
|---|---|---|---|---|---|
| Config format | YAML (jsonargparse) | YAML (jsonargparse) | YAML (OmegaConf) | Python/YAML | Pydantic |
| Torch-free merge | Yes | No | Yes (OmegaConf) | No | Yes |
| Multi-stage DAG | topology.py + dagster | No | No | No | Native |
| Sweep support | Recipe YAML expansion | None | `--multirun` | None | Partitions |
| Reproducibility | snapshot + hash + W&B | `SaveConfigCallback` | `outputs/` dir | `work_dir` dump | Asset metadata |
