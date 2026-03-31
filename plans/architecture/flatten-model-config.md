# Config Flatten — Complete Reference

> Completed: 2026-03-28 | Verified against source: 2026-03-30

Replaced Hydra/OmegaConf + config dataclasses with jsonargparse + flat YAML.
Every LightningModule `__init__` takes flat typed primitives. YAML is law.

## Current state

### Config package (`graphids/config/`)

Deleted (confirmed gone): `schema.py`, `coerce_config`, `resolve()`, `constants.py`, `defaults/`, `config.yaml`, `models.yaml`, `base.yaml`.

Kept:
- `__init__.py` — constants + topology + path helpers (`LAKE_ROOT`, `run_dir()`, `compute_identity_hash()`, `checkpoint_path()`)
- `pipeline.yaml` — DAG topology: stages, dependencies, identity_keys, valid models/scales
- `datasets.yaml` — dataset catalog (YAML anchors)
- `resources.yaml` — SLURM resource profiles per model×scale×stage

Added:
- `constants.yaml` — static values: preprocessing_version, SLURM defaults, ckpt_stages mapping
- `trainer.yaml` — shared trainer defaults (seed, precision, callbacks, loggers)
- `stages/` (9 files): `autoencoder.yaml`, `normal.yaml`, `curriculum.yaml`, `fusion.yaml`, `fusion_mlp.yaml`, `fusion_weighted_avg.yaml`, `analyze_vgae.yaml`, `analyze_gat.yaml`, `analyze_fusion.yaml`
- `overlays/` (7 files): `small_vgae.yaml`, `small_gat.yaml`, `small_dgi.yaml`, `large_vgae.yaml`, `large_gat.yaml`, `kd_vgae.yaml`, `kd_gat.yaml`
- `recipes/ablation.yaml` — 18 configs, claim-driven ablation recipe

### Model files (8 LightningModules)

| Module | File | Args | Prefix | Mixin |
|--------|------|------|--------|-------|
| `VGAEModule` | `vgae.py:332` | 28 | — | `OOMSkipMixin` |
| `GATModule` | `gat.py:183` | 28 | — | `OOMSkipMixin` |
| `DGIModule` | `dgi.py:149` | 20 | — | `OOMSkipMixin` |
| `TemporalLightningModule` | `temporal.py:146` | 25 | `spatial_*`, `temporal_*` | — |
| `DQNFusionModule` | `dqn.py` | (flat) | — | — |
| `BanditFusionModule` | `bandit.py` | (flat) | — | — |
| `MLPFusionModule` | `fusion_baselines.py:33` | 3 | — | — |
| `WeightedAvgModule` | `fusion_baselines.py:97` | 2 | — | — |
| `Analyzer` | `cli.py` | (varies) | `analyzer.*` | — |

Arg groups follow a common pattern across VGAE/GAT/DGI:
- **Architecture** — model-specific (conv_type, hidden_dims/hidden, heads, dropout, etc.)
- **Training** — lr, weight_decay, gradient_checkpointing, compile_model (+ loss_fn/focal_gamma/loss_weight for GAT)
- **Identity/dynamic** — scale, model_type, lake_root, dataset, seed, gat_stage, auxiliaries, num_ids, in_channels, num_classes

All `*Config` imports, `coerce_config` calls, and nested hparams access (`self.hparams.vgae.*`, `self.hparams.training.*`) eliminated.

### Inner nn.Module `from_config` methods

Updated to flat keys — these read flat `hparams` namespaces directly:
- `GraphAutoencoderNeighborhood.from_config` (`vgae.py`)
- `GATWithJK.from_config` (`gat.py`)
- `GraphInfomaxModel.from_config` (`dgi.py`)
- `DQNFusionModule` (`dqn.py`) — was `EnhancedDQNFusionAgent.from_config`, now proper LightningModule `__init__`
- `BanditFusionModule` (`bandit.py`) — was `NeuralLinUCBAgent.from_config`, now proper LightningModule `__init__`
- ~~`reward_kwargs_from_cfg` (`fusion_reward.py`)~~ — **deleted** (dead code, removed 2026-03-30)

`QNetwork` (`dqn.py:30`) takes plain args (`state_dim`, `action_dim`, `hidden_dim`, `num_layers`) — no `from_config` method.

### Structured validation

`KDAuxiliary` (`_training.py`) is a `TypedDict` — the only structured schema object. jsonargparse validates keys at parse time, catching typos like `alppha`.

```python
class KDAuxiliary(TypedDict, total=False):
    type: str
    alpha: float
    vgae_latent_weight: float   # VGAE KD only
    vgae_recon_weight: float    # VGAE KD only
    temperature: float          # GAT KD only
    teacher_scale: str
    model_path: str
```

### Supporting files

- `_training.py`: `cfg.vgae.latent_dim` → `cfg.latent_dim` in `prepare_kd`
- `__main__.py`: stale `TrainingConfig` comment cleaned

### Checkpoint migration

`scripts/migrate_checkpoints.py` — rewrites `hyper_parameters` in `.ckpt` files from nested to flat. Supports `--dry-run`. Run once against `experimentruns/` before training with new code.

```bash
python scripts/migrate_checkpoints.py experimentruns/ --dry-run
python scripts/migrate_checkpoints.py experimentruns/
```

## Non-Lightning nested references

- `curriculum.py:CurriculumSampler` — `CurriculumDataModule.setup()` builds a synthetic `SimpleNamespace(training=...)` internally. Not broken.

(`generate.py` and `registry._dqn_from_config` — referenced in original plan but no longer exist.)

## Verification

1. Import check: `python -c "from graphids.core.models.vgae import VGAEModule; print('OK')"`
2. Introspection: `python -m graphids fit --config graphids/config/stages/autoencoder.yaml --print_config`
3. Tests: `bash scripts/slurm/run_tests_slurm.sh`
4. Checkpoint round-trip: save → `load_from_checkpoint` → verify flat hparams
