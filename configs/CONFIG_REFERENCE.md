# Config Reference

> Audited: 2026-04-08. Parameter details live in jsonnet/libsonnet sources
> and Python `__init__` signatures — this doc covers conceptual structure
> and infrastructure that isn't obvious from reading the code.

---

## 1. Datasets

Catalog: `configs/datasets/dataset_registry.json`

| Dataset   | Source                   | Attack types                              |
|-----------|--------------------------|-------------------------------------------|
| `hcrl_ch` | HCRL Challenge (Korea U) | dos, fuzzing, gear_spoofing, rpm_spoofing |
| `hcrl_sa` | HCRL Scenario Anomaly    | mixed                                     |
| `set_01`–`set_04` | Automotive CAN  | mixed, suppress, masquerade               |

DataModules: `graphids/core/data/datamodule/graph.py` (GraphDataModule)
and `fusion.py` (FusionDataModule). No CurriculumDataModule — curriculum
was removed; GraphDataModule handles all non-fusion stages.

---

## 2. Model Parameters

Parameter values and scales live in jsonnet libsonnets — read those directly:

| Family       | Libsonnet                          | Module                                        |
|--------------|------------------------------------|-----------------------------------------------|
| Unsupervised | `configs/models/unsupervised.libsonnet` | `core/models/autoencoder/vgae_module.py`, `dgi_module.py` |
| Supervised   | `configs/models/supervised.libsonnet`   | `core/models/supervised/gat_module.py`   |
| Fusion       | `configs/fusion.libsonnet` + `configs/fusion/methods/*.libsonnet` | `core/models/fusion/{bandit,dqn,mlp,weighted_avg}.py` |

### Scale axis

Each libsonnet defines `base` (shared) + `scales.small` / `scales.large`.
Stage jsonnet merges `base + scales[scale]`. Key dimensions that vary by scale:
hidden dims, latent dim, heads, dropout, proj_dim, fc_layers.

### Fusion methods

4 methods: `bandit`, `dqn`, `mlp`, `weighted_avg`. Method-specific params
(buffer_size, epsilon, ucb_alpha, etc.) live in
`configs/fusion/methods/{method}.libsonnet`. Shared trainer config
(precision=32, max_epochs=50, monitor=val_acc/max) in `configs/fusion/base.libsonnet`.

Reward shaping constants are fixed module-level constants in
`core/models/fusion/fusion_reward.py` — not tunable via config.
Only `vgae_weights` in `reward_kwargs` is configurable.

### KD auxiliaries

Schema: `KDEntry` in `graphids/orchestrate/config.py`.
Fields: `type`, `alpha`, `temperature`, `model_path`,
`vgae_latent_weight`, `vgae_recon_weight`.

---

## 3. Trainer Defaults

Defaults: `configs/_lib/defaults.libsonnet` (trainer, checkpoint, early_stopping).

| Setting                | AE / GAT       | Fusion                    |
|------------------------|----------------|---------------------------|
| `precision`            | `16-mixed`     | `32` (manual opt)         |
| `max_epochs`           | `300`          | `50`                      |
| `gradient_clip_val`    | `1.0`          | `null`                    |
| `checkpoint.monitor`   | `val_loss/min` | `val_acc/max`             |
| `early_stopping`       | `val_loss/min, patience=100` | `val_acc/max`  |
| `DynamicBatchSampler`  | active         | inactive                  |

### LR schedulers (in code, not config)

| Module          | Optimizer | Scheduler                |
|-----------------|-----------|--------------------------|
| VGAE/GAT/DGI   | Adam      | CosineAnnealingLR        |
| DQN/Bandit      | Adam      | none                     |
| MLP/WeightedAvg | Adam      | none                     |

### Forced callbacks

`ModelCheckpoint` and `EarlyStopping` are declared in `defaults.libsonnet`.
`OTelTrainingCallback` replaces the former DeviceStatsMonitor and ResourceProfileCallback.

---

## 4. Resources

### Environment variables

Project `.env` (sourced by `_preamble.sh`):

| Variable                 | Purpose                          |
|--------------------------|----------------------------------|
| `GRAPHIDS_LAKE_ROOT`      | ESS data lake root               |
| `GRAPHIDS_SLURM_ACCOUNT`  | SLURM account (PAS1266)          |
| `GRAPHIDS_SLURM_LOG_DIR`  | SLURM log directory              |
| `GRAPHIDS_SCRATCH`        | Scratch filesystem root          |
| `GRAPHIDS_DATA_ROOT`      | Raw data directory               |
| `GRAPHIDS_LAKE_WRITE`     | Write guard for ESS (1=enabled)  |
| `GRAPHIDS_CLUSTER`        | Override auto-detected cluster   |
| `GRAPHIDS_DRY_RUN`        | Skip sbatch (1=dry run)          |

Python reads: `graphids/config/constants.py` and `graphids/slurm/env.py`.
Budget tuning: `GRAPHIDS_BUDGET_SAFETY_MARGIN`, `GRAPHIDS_BUDGET_GRAD_MULT`,
`GRAPHIDS_BUDGET_FALLBACK_BPN` in `core/data/budget.py`.

### HPC resource profiles

Single source of truth: `configs/resources/submit_profiles.json`. Three shapes:

- **static** — fixed `time` + `mem` (e.g. `tests`, `analyze`, `profile`).
- **scaling** — `scaling.{time_min, mem_gb}` with `base + per_mraw * num_raw_samples`
  interpolated from `cache_metadata.json.aggregate.num_raw_samples` for the
  `--dataset` passed to `submit.sh`. Falls back to the `defaults` block when
  `--dataset` is absent. `rebuild-caches` uses this.
- **composed** — `stages: [<stage_profile_key>, ...]` pulling from top-level
  `stage_profiles`. `time = sum(stages)`, `cpus/mem = max(stages)`. Scale
  (small/large) applied via `scale_mult`. `pipeline-run` uses this.

`submit-profile <job> [--dataset X] [--scale small|large]` prints the resolved
row that `submit.sh` parses. `graphids/config/topology.py::_validate_submit_profiles`
catches unknown stage references and out-of-range scale keys at import time.

### Submit profiles (`scripts/slurm/submit.sh`)

| Profile        | Partition | Time  | Command                               |
|----------------|-----------|-------|---------------------------------------|
| tests          | cpu       | 1:00  | `python -m pytest`                    |
| rebuild-caches | cpu       | 4:00  | `python -m graphids rebuild-caches`   |
| profile        | gpudebug  | 1:00  | `python -m graphids profile`          |

Full list: `configs/resources/submit_profiles.json`.

---

## 5. Storage & IO

### Storage tiers

| Tier             | Path                          | Persistence  | Use                          |
|------------------|-------------------------------|--------------|------------------------------|
| NFS (home)       | `~/graphids/data/`           | Permanent    | Raw data source of truth     |
| ESS (GPFS)       | `/fs/ess/PAS1266/graphids/`  | Permanent    | Lake root: runs, catalog     |
| Scratch (GPFS)   | `/fs/scratch/PAS1266/`       | 90-day purge | wandb, data staging          |
| TMPDIR (local)   | `$TMPDIR/graphids-data/`     | Per-job      | Training I/O                 |

### Run directory template

```
{lake_root}/{production|dev/user}/{dataset}/{family}_{scale}_{stage}_{identity_hash}/seed_{N}
```

Identity hash: 8-char SHA256 from stage identity keys (defined in `topology.py`).
Computed by `compute_identity_hash()` in `graphids/config/paths.py`.

### Logged metrics

| Model       | train step            | val step        | test epoch                                        |
|-------------|-----------------------|-----------------|---------------------------------------------------|
| VGAE/DGI    | train_loss            | val_loss        | accuracy, f1, precision, recall, specificity, auc |
| GAT         | train_loss, train_acc | val_loss, val_acc | accuracy, f1, precision, recall, specificity, auc |
| DQN         | avg_reward, epsilon   | val_acc         | --                                                |
| Bandit      | accuracy, avg_reward  | val_acc         | --                                                |
| MLP/WAvg    | train_loss            | val_loss, val_acc | accuracy, f1, precision, recall, specificity, auc |

### Analyzer artifacts

`ARTIFACTS_BY_MODEL_TYPE` in `core/analysis/schemas.py` dispatches by the
checkpoint's self-describing `class_path` — both `analyze` CLI and the
pipeline driver go through `analysis_spec_for`, so the toggles below fire
automatically without a per-run config.

| model_type | embeddings | attention | cka | landscape   | fusion_policy |
|------------|------------|-----------|-----|-------------|---------------|
| `vgae`     | yes        | --        | --  | yes (51x51) | --            |
| `dgi`      | yes        | --        | --  | yes (51x51) | --            |
| `gat`      | yes        | yes       | yes | yes         | --            |
| `fusion`   | --         | --        | --  | --          | yes (needs upstream ckpts) |

| Artifact       | File                                  | Contents                            |
|----------------|---------------------------------------|-------------------------------------|
| Embeddings     | `embeddings.npz`                      | embeddings, labels, model_type      |
| Attention      | `attention_weights.npz`               | per-sample per-layer alpha weights  |
| CKA            | `cka.json`                            | per-layer student/teacher similarity|
| Landscape      | `loss_landscape_{model_type}.parquet` | x, y, loss grid                     |
| Fusion policy  | `dqn_policy.json`                     | alphas, labels, q_values            |

### Data I/O

Jobs read raw CSVs and cached tensors directly from ESS NFS
(`/fs/ess/PAS1266/graphids/{raw,cache}/`). No scratch/TMPDIR staging
today — the old `stage-data` command was removed 2026-04-14.
