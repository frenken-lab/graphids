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

Reward shaping constants live in `configs/models/fusion/reward.libsonnet` —
imported by `dqn.libsonnet` and `bandit.libsonnet`, all values flow into
`reward_kwargs`. They are fixed methodological choices (not ablation axes)
but are config-tunable.

### KD auxiliaries

Jsonnet TLA `distillation_config` (a dict) flows into
`model.init_args.distillation_config`; `inject_loss_fn` in
`graphids/core/losses/build.py` pops it and wraps the base loss with
`SoftLabelDistillation` (GAT) or `FeatureDistillation` (VGAE) from
`core/losses/distillation.py`. Fields: `type`, `alpha`, `temperature`,
`model_path`, `vgae_latent_weight`, `vgae_recon_weight`.

---

## 3. Trainer Defaults

Defaults: `configs/_lib/defaults.libsonnet` (trainer, checkpoint, early_stopping).

| Setting                | AE (VGAE)                             | GAT                      | Fusion               |
|------------------------|---------------------------------------|--------------------------|----------------------|
| `precision`            | `32-true`                             | `32-true`                | `32-true`            |
| `max_epochs`           | `600`                                 | `200`                    | `1500`               |
| `gradient_clip_val`    | `1.0`                                 | `1.0`                    | `null`               |
| `checkpoint.monitor`   | `val_discrimination_ratio/max`        | `val_loss/min`           | `val_acc/max`        |
| `early_stopping`       | `val_discrimination_ratio/max, p=100` | `val_loss/min, p=30`     | `val_acc/max, p=200` |
| `DynamicBatchSampler`  | active                                | active                   | inactive             |

### LR schedulers (in code, not config)

| Module          | Optimizer | Scheduler                |
|-----------------|-----------|--------------------------|
| VGAE/GAT/DGI   | Adam      | CosineAnnealingLR        |
| DQN/Bandit      | Adam      | none                     |
| MLP/WeightedAvg | Adam      | none                     |

### Forced callbacks

`ModelCheckpoint` and `EarlyStopping` are declared in `defaults.libsonnet`.
`MLflowTrainingCallback` (`graphids/core/mlflow_callback.py`) logs per-epoch metrics + fit-end peak VRAM to MLflow. Device telemetry is captured by MLflow's background system-metrics sampler (psutil + nvidia-ml-py, 5s interval).

---

## 4. Resources

### Environment variables

Project `.env` (sourced by `_preamble.sh`):

| Variable                 | Purpose                          |
|--------------------------|----------------------------------|
| `GRAPHIDS_LAKE_ROOT`      | ESS data lake root (shared: mlflow.db, cache, mlartifacts) |
| `GRAPHIDS_RUN_ROOT`       | Per-user run root (run_dirs / checkpoints) — `${LAKE_ROOT}/dev/${USER}` on OSC |
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

Single source of truth: `configs/resources/submit_profiles.json`. Exactly two
entries — `gpu` and `cpu`. Each carries per-cluster `partitions` and per-length
`times` defaults. Per-job mem/time/command are flags on `python -m graphids
submit`, never new JSON entries. `graphids/slurm/submit.py` loads the profiles at submission time.

Optional MLflow-history walltime estimation lives in
`graphids.slurm.sizing.estimate_walltime_minutes`; `python -m graphids submit
--time-from-history` opts into it for fit jobs with enough history (≥3 prior
FINISHED runs).

### Submission surface

| Use | Command |
|-----|---------|
| Training preset | `python -m graphids submit <preset.jsonnet> [--dataset X --seed N --smoke]` |
| Test / eval (CPU) | `python -m graphids submit --mode cpu --command "python -m graphids test --config X --ckpt Y"` |
| Tests | `python -m graphids submit --mode cpu --length short --command "python -m pytest [-k pattern]"` |
| Cache rebuild | `python -m graphids submit --mode cpu --mem-gb 54 --timeout-min 240 --command "python -m graphids rebuild-caches --all"` |
| Analyze ckpt | `python -m graphids submit --mode gpu --mem-gb 32 --timeout-min 120 --command "python -m graphids analyze ..."` |
| Profiling | `python -m graphids submit --mode gpu --length short --command "python -m graphids profile"` |

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

Every preset under `configs/ablations/*.jsonnet` computes its own
`run_dir` via `std.native('paths.run_dir')(...)` — registered by
`graphids.config.jsonnet.render()` against `graphids.config.paths.run_dir`:

```
{run_root}/{dataset}/ablations/{group}/{variant}/seed_{N}
```

No Python planner, no identity-hash layer.

### Logged metrics

Classifier-flavor models (GAT, all fusion) emit the unified
`classification_test_metrics` set on `test_epoch`: `accuracy`, `mcc`, `ece`;
`{f1,precision,recall,specificity,auc,ap}_{macro,weighted}`; and per-class
`{f1,precision,recall,specificity,auc,ap}_per_class/<name>` via
`torchmetrics.wrappers.ClasswiseWrapper` (class names default to
`["benign","attack"]` for binary). Threshold-flavor models (VGAE/DGI) keep
`binary_test_metrics(threshold=Youden-J)`: `accuracy, f1, precision, recall,
specificity, mcc, auc, ap, ece` plus the discovered `threshold`.

| Model       | train step            | val step          | test epoch |
|-------------|-----------------------|-------------------|-----------|
| VGAE / DGI  | `train_loss`          | `val_loss`        | binary @ Youden-J |
| GAT         | `train_loss, train_acc` | `val_loss, val_acc` | classifier (unified) |
| DQN         | `avg_reward, epsilon` | `val_acc`         | classifier (unified) |
| Bandit      | `accuracy, avg_reward`| `val_acc`         | classifier (unified) |
| MLP / WAvg  | `train_loss`          | `val_loss, val_acc` | classifier (unified) |

### Analyzer artifacts

`ARTIFACTS_BY_MODEL_TYPE` in `core/analysis/schemas.py` dispatches by the
checkpoint's self-describing `class_path` — the `analyze` CLI reads the
ckpt, looks up the spec via `analysis_spec_for`, and fires the toggles
below automatically without a per-run config.

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
