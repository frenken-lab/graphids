# Config Reference

> Reorganized: 2026-03-31 — merged from `PARAMETER_AXES.md` + `INFRASTRUCTURE_REFERENCE.md`.
> Grouped by config domain. For axis classifications (scale-only, model-only, etc.) see `PARAMETER_AXES.md`.

---

## 1. Dataset Configs

### Declared datasets (`datasets.yaml`)

| Dataset   | Source                     | Attack types                                      |
|-----------|----------------------------|---------------------------------------------------|
| `hcrl_ch` | HCRL Challenge (Korea U)   | dos, fuzzing, gear_spoofing, rpm_spoofing         |
| `hcrl_sa` | HCRL Scenario Anomaly      | mixed                                             |
| `set_01`  | Automotive CAN Set 01      | mixed, suppress, masquerade                       |
| `set_02`  | Automotive CAN Set 02      | mixed, suppress, masquerade                       |
| `set_03`  | Automotive CAN Set 03      | mixed, suppress, masquerade                       |
| `set_04`  | Automotive CAN Set 04      | mixed, suppress, masquerade                       |

Full dataset specs (csv_columns, test_subdirs, attack_types) live in `datasets.yaml`.

### Preprocessing constants (`core/preprocessing/features.py`)

| Constant           | Value | Location         | Note                                                 |
|--------------------|-------|------------------|------------------------------------------------------|
| `N_NODE_FEATURES`  | 43    | `features.py:55` | Fixed across all stages                              |
| `N_EDGE_FEATURES`  | 11    | `features.py:64` | Comment in file says 10 — stale, not a bug           |
| `window_size`      | 100   | DataModule default |                                                    |
| `stride`           | 100   | DataModule default |                                                    |

Changing any of these invalidates all cached graphs (used in `compute_preprocessing_hash()`).

### DataModule parameters

#### `CANBusDataModule` — autoencoder, normal stages (`core/preprocessing/datamodule.py:205`)

| Parameter          | Type + Default      | Stage YAML value | Axis                                 |
|--------------------|---------------------|------------------|--------------------------------------|
| `dataset`          | `str` (required)    | set per run      | runtime                              |
| `batch_size`       | `int = 32`          | `8192`           | stage-default (**dead** when `dynamic_batching=True`) |
| `num_workers`      | `int = 2`           | `2`              | stage-default                        |
| `window_size`      | `int = 100`         | —                | fixed                                |
| `stride`           | `int = 100`         | —                | fixed                                |
| `val_fraction`     | `float = 0.2`       | —                | fixed                                |
| `seed`             | `int = 42`          | —                | fixed                                |
| `dynamic_batching` | `bool = True`       | —                | fixed (disable with `-O`)            |
| `conv_type`        | `str = "gatv2"`     | —                | pass-through (VRAM budget probe)     |
| `heads`            | `int = 4`           | —                | pass-through (VRAM budget probe)     |

#### `CurriculumDataModule` — curriculum stage (`core/preprocessing/curriculum.py:106`)

Extends `CANBusDataModule`, adds:

| Parameter                 | Type + Default   | Stage YAML value | Axis               |
|---------------------------|------------------|------------------|--------------------|
| `vgae_ckpt_path`          | `str = ""`       | set per run      | runtime            |
| `curriculum_start_ratio`  | `float = 1.0`    | —                | curriculum tuning  |
| `curriculum_end_ratio`    | `float = 10.0`   | —                | curriculum tuning  |
| `difficulty_percentile`   | `float = 75.0`   | —                | curriculum tuning  |
| `canid_weight`            | `float = 0.1`    | —                | curriculum tuning  |
| `max_epochs`              | `int = 300`      | —                | **must match `trainer.max_epochs`** |

#### `FusionDataModule` — fusion stages (`core/preprocessing/datamodule.py:327`)

| Parameter             | Type + Default    | Stage YAML value    | Axis                              |
|-----------------------|-------------------|---------------------|-----------------------------------|
| `vgae_ckpt_path`      | `str = ""`        | set per run         | runtime                           |
| `gat_ckpt_path`       | `str = ""`        | set per run         | runtime                           |
| `method`              | `str = "bandit"`  | set by method YAML  | method-only                       |
| `batch_size`          | `int = 128`       | —                   | **ignored for RL methods**        |
| `episode_sample_size` | `int = 20000`     | —                   | method-only (bandit/dqn only)     |
| `max_samples`         | `int = 150000`    | —                   | fixed                             |
| `max_val_samples`     | `int = 30000`     | —                   | fixed                             |
| `eval_batch_size`     | `int = 256`       | —                   | fixed                             |

---

## 2. Model Configs

### VGAE / DGI — autoencoder stage

Source: `VGAEModule` (`core/models/vgae.py:340`), `DGIModule` (`core/models/dgi.py:155`)

| Parameter                  | Type + Default              | small         | large        | Axis        |
|----------------------------|-----------------------------|---------------|--------------|-------------|
| `hidden_dims`              | `list[int] \| None = None`  | [80, 40, 16]  | [480,240,64] | scale-only  |
| `latent_dim`               | `int = 48`                  | 16            | 64           | scale-only  |
| `heads`                    | `int = 4`                   | 1             | 4            | scale-only  |
| `embedding_dim`            | `int = 32`                  | 4             | 32           | scale-only  |
| `dropout`                  | `float = 0.15`              | 0.1           | 0.15         | scale-only  |
| `scale`                    | `str = "small"`             | (default)     | "large"      | scale-only  |
| **`proj_dim`**             | **`int = 0`**               | **32**        | **48**       | **coupled** (stage=48, small overrides to 32) |
| `variational`              | `bool = True`               | —             | —            | model-only (VGAE only) |
| `mask_ratio`               | `float = 0.3`               | —             | —            | model-only (VGAE only) |
| `k_neg`                    | `int = 32`                  | —             | —            | model-only (VGAE only) |
| `canid_weight`             | `float = 0.1`               | —             | —            | model-only (VGAE only) |
| `nbr_weight`               | `float = 0.05`              | —             | —            | model-only (VGAE only) |
| `kl_weight`                | `float = 0.01`              | —             | —            | model-only (VGAE only) |
| `lr`                       | `float = 0.003`             | —             | —            | stage-default (autoencoder.yaml sets 0.002) |
| `compile_model`            | `bool = False`              | —             | —            | stage-default (autoencoder.yaml sets true) |
| `conv_type`                | `str = "gatv2"`             | —             | —            | stage-default |
| `edge_dim`                 | `int = 11`                  | —             | —            | stage-default |
| `gradient_checkpointing`   | `bool = True`               | —             | —            | stage-default |
| `auxiliaries`              | `list[KDAuxiliary] \| None` | —             | —            | stage-default (autoencoder.yaml sets []) |

**DGI note:** shares VGAE architecture params but lacks `variational`, `mask_ratio`, `k_neg`, `canid_weight`, `nbr_weight`, `kl_weight`, `lr`, `weight_decay`. No `dgi/large.yaml` — large DGI is undeclared.

### GAT — normal / curriculum stages

Source: `GATModule` (`core/models/gat.py:197`)

| Parameter        | Type + Default              | small     | large | Axis        |
|------------------|-----------------------------|-----------|-------|-------------|
| `hidden`         | `int = 48`                  | 24        | 64    | scale-only  |
| `layers`         | `int = 3`                   | 2         | 3     | scale-only  |
| `heads`          | `int = 8`                   | 4         | 4     | scale-only  |
| `embedding_dim`  | `int = 16`                  | 8         | 8     | scale-only  |
| `dropout`        | `float = 0.2`               | 0.1       | 0.11  | scale-only  |
| `scale`          | `str = "small"`             | (default) | "large" | scale-only |
| **`proj_dim`**   | **`int = 0`**               | **32**    | **48** | **coupled** (stage=48, small overrides to 32) |
| **`fc_layers`**  | **`int = 3`**               | **2**     | **4** | **coupled** (stage=1, both scales override differently) |
| `pool_aggrs`     | `list[str] \| None`         | —         | —     | stage-default (normal.yaml sets ["mean"]) |
| `compile_model`  | `bool = False`              | —         | —     | stage-default (normal.yaml sets true) |
| `auxiliaries`    | `list[KDAuxiliary] \| None` | —         | —     | stage-default (normal.yaml sets []) |
| `conv_type`      | `str = "gatv2"`             | —         | —     | stage-default |
| `loss_fn`        | `str = "ce"`                | —         | —     | stage-default |
| `focal_gamma`    | `float = 2.0`               | —         | —     | stage-default |
| `loss_weight`    | `float = 10.0`              | —         | —     | stage-default |

**normal.yaml vs curriculum.yaml:** Identical `model.init_args`. Only `data.class_path` differs (`CANBusDataModule` vs `CurriculumDataModule`).

### Fusion Methods

Source: `BanditFusionModule` (`core/models/fusion/bandit.py`), `DQNFusionModule` (`core/models/fusion/dqn.py`),
`MLPFusionModule` (`core/models/fusion/fusion_baselines.py`), `WeightedAvgModule` (`core/models/fusion/fusion_baselines.py`)

#### Shared trainer config (`fusion.yaml`)

`trainer.precision=32`, `trainer.max_epochs=50`, `trainer.gradient_clip_val=null`, `checkpoint.monitor=val_acc`, `checkpoint.mode=max`, `data.class_path=FusionDataModule`

#### Per-method parameters

| Parameter               | Bandit | DQN   | MLP      | WeightedAvg | Axis           |
|-------------------------|--------|-------|----------|-------------|----------------|
| `hidden_dim`            | 128    | 128   | —        | —           | scale-only     |
| `hidden_dims`           | —      | —     | (64, 32) | —           | scale-only     |
| `num_layers`            | 3      | 3     | —        | —           | scale-only     |
| `buffer_size`           | 100k   | 50k   | —        | —           | method-only    |
| `batch_size`            | 128    | 128   | —        | —           | scale-only     |
| `lr`                    | 0.001  | 0.001 | 0.001    | 0.01        | method-only    |
| `decision_threshold`    | 0.5    | 0.5   | —        | 0.5         | shared default |
| `reward_kwargs.vgae_weights` | `[0.4,0.3,0.3]` | `[0.4,0.3,0.3]` | — | — | method-only |
| `ucb_alpha`             | 1.0    | —     | —        | —           | bandit-only    |
| `lambda_reg`            | 1.0    | —     | —        | —           | bandit-only    |
| `backbone_lr`           | 0.001  | —     | —        | —           | bandit-only    |
| `backbone_retrain_freq` | 50     | —     | —        | —           | bandit-only    |
| `backbone_epochs`       | 5      | —     | —        | —           | bandit-only    |
| `epsilon`               | —      | 0.2   | —        | —           | dqn-only       |
| `epsilon_decay`         | —      | 0.995 | —        | —           | dqn-only       |
| `min_epsilon`           | —      | 0.01  | —        | —           | dqn-only       |
| `gpu_training_steps`    | —      | 1     | —        | —           | dqn-only       |
| `weight_decay`          | —      | 1e-5  | —        | —           | dqn-only       |

**Fusion YAML exposure gaps:** `dqn/small.yaml` and `dqn/large.yaml` are both `{}`. All fusion architecture params live as code defaults with zero YAML surface. No `bandit/`, `mlp/`, or `weighted_avg/` model config dirs exist.

**Reward shaping constants are fixed, not tunable.** The 7 coefficients used by `FusionRewardCalculator.compute()` (`_REWARD_CORRECT=±3.0`, `_CONFIDENCE_WEIGHT=0.5`, `_COMBINED_CONF_WEIGHT=0.3`, `_DISAGREEMENT_PENALTY=-1.0`, `_OVERCONF_PENALTY=-1.5`, `_BALANCE_WEIGHT=0.3`) are module-level constants in `core/models/fusion/fusion_reward.py`, matching the paper's `methodology.md §Stage 3` equation. They are not exposed as kwargs and are not ablation axes — DQN and bandit share the identical reward by design. `vgae_weights` is the only tunable in `reward_kwargs`.

### Temporal Model (not yet in pipeline)

Source: `TemporalLightningModule` (`core/models/temporal.py:158`) — no stage YAML.

| Parameter             | Type + Default          | Group                   |
|-----------------------|-------------------------|-------------------------|
| `spatial_hidden`      | `int = 48`              | spatial (GAT backbone)  |
| `spatial_layers`      | `int = 3`               | spatial                 |
| `spatial_heads`       | `int = 8`               | spatial                 |
| `spatial_dropout`     | `float = 0.2`           | spatial                 |
| `spatial_embedding_dim` | `int = 16`            | spatial                 |
| `spatial_conv_type`   | `str = "gatv2"`         | spatial                 |
| `spatial_edge_dim`    | `int = 11`              | spatial                 |
| `spatial_pool_aggrs`  | `list[str] \| None = None` | spatial              |
| `spatial_proj_dim`    | `int = 0`               | spatial                 |
| `spatial_fc_layers`   | `int = 3`               | spatial                 |
| `window_size`         | `int = 10`              | temporal sequence       |
| `stride`              | `int = 1`               | temporal sequence       |
| `temporal_hidden`     | `int = 64`              | temporal transformer    |
| `temporal_heads`      | `int = 4`               | temporal transformer    |
| `temporal_layers`     | `int = 2`               | temporal transformer    |
| `freeze_spatial`      | `bool = True`           | training                |
| `spatial_lr_factor`   | `float = 0.01`          | training                |
| `lr`                  | `float = 0.001`         | training                |
| `gat_ckpt_path`       | `str \| None = None`    | runtime (excluded from hparams) |

### KD Auxiliary parameters

Source: `KDAuxiliary` TypedDict (`core/models/_training.py:92`). All fields `total=False`. Set in `models/{model}/kd.yaml`.

| Key                  | Type    | Used by | Purpose                                 |
|----------------------|---------|---------|-----------------------------------------|
| `type`               | `str`   | all     | Discriminator — `"kd"` activates distillation |
| `alpha`              | `float` | all     | KD loss weight                          |
| `temperature`        | `float` | GAT     | Softmax temperature for logit matching  |
| `vgae_latent_weight` | `float` | VGAE    | Weight for latent-space alignment       |
| `vgae_recon_weight`  | `float` | VGAE    | Weight for reconstruction loss          |
| `teacher_config`     | `str`   | orchestration | **Required for pipeline runs.** Names the recipe config that produces the teacher — wired as an explicit upstream dagster dep by planning. |
| `teacher_scale`      | `str`   | dev-path only | Scale used by `prepare_kd` when running `python -m graphids fit` without the orchestrator. Ignored by planning. |
| `model_path`         | `str`   | all     | Explicit teacher ckpt path (overrides `teacher_scale`) |

---

## 3. Trainer / Training Strategy Configs

### Shared defaults (`trainer.yaml`)

| Key                        | Value      | Notes              |
|----------------------------|------------|--------------------|
| `seed_everything`          | `42`       | global seed        |
| `trainer.accelerator`      | `auto`     |                    |
| `trainer.devices`          | `auto`     |                    |
| `trainer.precision`        | `16-mixed` | fp16 AMP           |
| `trainer.max_epochs`       | `300`      |                    |
| `trainer.gradient_clip_val`| `1.0`      |                    |
| `trainer.log_every_n_steps`| `50`       |                    |

### Per-stage overrides

| Parameter             | AE / GAT            | Fusion               | Axis       |
|-----------------------|---------------------|----------------------|------------|
| `precision`           | `16-mixed` (inherit)| `32` (AMP incompatible with manual opt) | stage |
| `max_epochs`          | `300` (inherit)     | `50`                 | stage      |
| `gradient_clip_val`   | `1.0` (inherit)     | `null` (manual opt)  | stage      |
| `log_every_n_steps`   | `50` (inherit)      | `10`                 | stage      |
| `checkpoint.monitor`  | `val_loss` / min    | `val_acc` / max      | stage      |
| `early_stopping.monitor` | `val_loss` / min | `val_acc` / max      | stage      |

### Training strategy settings per stage

| Parameter                 | AE / GAT              | Fusion (DQN/Bandit)         | Fusion (MLP/WAvg) |
|---------------------------|-----------------------|-----------------------------|-------------------|
| `data.batch_size`         | `8192` (dead*)        | `episode_sample_size=20000` | `128`             |
| `data.num_workers`        | `2`                   | `0`                         | `0`               |
| `persistent_workers`      | `True` (hardcoded)    | N/A                         | N/A               |
| `trainer.precision`       | `16-mixed`            | `32`                        | `32`              |
| `trainer.gradient_clip_val` | `1.0`               | `null`                      | `null`            |
| `model.compile_model`     | `true`                | `false` (default)           | `false` (default) |
| `model.gradient_checkpointing` | `true` (default) | N/A                        | N/A               |
| `DynamicBatchSampler`     | active (node budget)  | inactive                    | inactive          |
| LR scheduler              | `CosineAnnealingLR`   | none                        | none              |

*`batch_size=8192` is dead config when `dynamic_batching=True` (default). `DynamicBatchSampler` bypasses it entirely.

### LR schedulers (all in code, none in YAML)

| Module      | Optimizer          | Scheduler                               |
|-------------|--------------------|-----------------------------------------|
| VGAE        | Adam               | `CosineAnnealingLR(T_max=max_epochs)`   |
| GAT         | Adam               | `CosineAnnealingLR` (inherited)         |
| DGI         | Adam               | `CosineAnnealingLR` (inherited)         |
| Temporal    | AdamW (split param groups) | none                            |
| DQN         | Adam (pre-built)   | none                                    |
| Bandit      | Adam (pre-built)   | none                                    |
| MLP         | Adam               | none                                    |
| WeightedAvg | Adam               | none                                    |

### Callbacks

#### Force-registered (`cli.py` — immune to YAML list replacement)

| Callback            | Namespace          | Defaults                                         | Stage overrides                     |
|---------------------|--------------------|--------------------------------------------------|-------------------------------------|
| `ModelCheckpoint`   | `checkpoint.*`     | `monitor=val_loss`, `mode=min`, `save_top_k=1`, `save_last=true`, `filename=best_model` | Fusion: `monitor=val_acc`, `mode=max` |
| `EarlyStopping`     | `early_stopping.*` | `monitor=val_loss`, `patience=100`, `mode=min`   | Fusion: `monitor=val_acc`, `mode=max` |

#### In `trainer.callbacks` list (`trainer.yaml`)

| Callback             | Purpose                  |
|----------------------|--------------------------|
| `DeviceStatsMonitor` | GPU utilization logging  |

### CLI argument linking (`cli.py:GraphIDSCLI.add_arguments_to_parser()`)

| Source                       | Target                       | Purpose                                 |
|------------------------------|------------------------------|-----------------------------------------|
| `data.init_args.dataset`     | `model.init_args.dataset`    | Model gets dataset name for path computation |
| `data.init_args.lake_root`   | `model.init_args.lake_root`  | Model gets lake root for checkpoint paths |
| `seed_everything`            | `model.init_args.seed`       | Model gets seed for run_dir             |
| `seed_everything`            | `data.init_args.seed`        | DataModule gets seed for splits         |
| `model.init_args.conv_type`  | `data.init_args.conv_type`   | DataModule VRAM budget probe            |
| `model.init_args.heads`      | `data.init_args.heads`       | DataModule VRAM budget probe            |

---

## 4. Resource Configs

### Environment variables

#### Project `.env` (sourced by `_preamble.sh` and `submit.sh`)

| Variable                  | Value                                   | Consumed by                             |
|---------------------------|-----------------------------------------|-----------------------------------------|
| `KD_GAT_SLURM_ACCOUNT`    | `PAS1266`                               | `config/__init__.py`, `slurm.py`, `submit.sh` |
| `KD_GAT_SLURM_PARTITION`  | `gpu`                                   | `config/__init__.py`                    |
| `KD_GAT_GPU_TYPE`         | `v100`                                  | `config/__init__.py`                    |
| `KD_GAT_LAKE_ROOT`        | `/fs/ess/PAS1266/kd-gat`                | `config/__init__.py`                    |
| `KD_GAT_SLURM_LOG_DIR`    | `/fs/ess/PAS1266/kd-gat/slurm_logs`    | `config/__init__.py`, `slurm.py`        |
| `KD_GAT_SCRATCH`          | `/fs/scratch/PAS1266`                   | `.env` only                             |
| `KD_GAT_PYTHON`           | `python`                                | `.env` only (legacy)                    |
| `KD_GAT_SHARED_ROOT`      | `/fs/scratch/PAS1266/kd-gat-shared`     | `.env` only                             |
| `KD_GAT_DATA_ROOT`        | `/users/PAS2022/rf15/KD-GAT/data`       | `.env` only (local fallback)            |
| `DAGSTER_HOME`            | `/fs/scratch/PAS1266/dagster`           | dagster (SQLite event log)              |

#### Runtime env vars (set by `_preamble.sh`)

| Variable                   | Value                                                       | Purpose                        |
|----------------------------|-------------------------------------------------------------|--------------------------------|
| `WANDB_DIR`                | read from `config.WANDB_WRITE_DIR`                          | wandb scratch writes           |
| `WANDB_DISABLE_GIT`        | `true`                                                      | skip NFS git probe             |
| `WANDB_SILENT`             | `true`                                                      | reduce SLURM log noise         |
| `PYTORCH_CUDA_ALLOC_CONF`  | `expandable_segments:True,garbage_collection_threshold:0.8` | CUDA allocator (skipped for CPU jobs) |
| `KD_GAT_STAGE_DIR`         | `$TMPDIR/kd-gat-stage`                                      | node-local staging scratch     |

#### Python-read env vars (`config/__init__.py`)

| Variable               | Fallback                           | Python constant    |
|------------------------|------------------------------------|--------------------|
| `KD_GAT_SLURM_ACCOUNT` | `constants.yaml → PAS1266`         | `SLURM_ACCOUNT`    |
| `KD_GAT_SLURM_LOG_DIR` | `constants.yaml → /fs/ess/...`     | `SLURM_LOG_DIR`    |
| `KD_GAT_LAKE_ROOT`     | `"experimentruns"` (relative)      | `LAKE_ROOT`        |
| `KD_GAT_SLURM_PARTITION` | `constants.yaml → gpu`           | `SLURM_PARTITION`  |
| `KD_GAT_GPU_TYPE`      | `constants.yaml → v100`            | `SLURM_GPU_TYPE`   |
| `KD_GAT_SWEEP_ID`      | `""`                               | `SWEEP_ID`         |
| `KD_GAT_TAGS`          | `""`                               | `USER_TAGS`        |
| `KD_GAT_CKPT_PATH`     | `""`                               | `CKPT_PATH`        |
| `WANDB_DIR`            | `write_paths.yaml → /fs/scratch/PAS1266/wandb` | `WANDB_WRITE_DIR` |
| `KD_GAT_CLUSTER`       | hostname detection                 | `resources.py:_detect_cluster()` |
| `KD_GAT_RECIPE`        | `recipes/ablation.yaml`            | `component.py:RECIPE_PATH` |

`env_prefix="KD_GAT"` in `CLI_KWARGS` — any `__init__` param settable via `KD_GAT_MODEL__INIT_ARGS__LR=0.001`.

### Static constants (`constants.yaml`)

| Constant              | Value                     | Python symbol           | Purpose                     |
|-----------------------|---------------------------|-------------------------|-----------------------------|
| `preprocessing_version` | `"7.0.0"`               | `PREPROCESSING_VERSION` | Cache invalidation key      |
| `max_data_bytes`      | `8`                       | `MAX_DATA_BYTES`         | CAN data field width        |
| `excluded_attack_types` | `[suppress, masquerade]`| `EXCLUDED_ATTACK_TYPES`  | Filtered from training      |
| `slurm.account`       | `PAS1266`                 | fallback for `SLURM_ACCOUNT` |                        |
| `slurm.partition`     | `gpu`                     | fallback for `SLURM_PARTITION` |                      |
| `slurm.gpu_type`      | `v100`                    | fallback for `SLURM_GPU_TYPE` |                       |
| `slurm.log_dir`       | `/fs/ess/PAS1266/kd-gat/slurm_logs` | fallback for `SLURM_LOG_DIR` |           |

### HPC resource profiles (`resources.yaml`)

#### Cluster mapping

| Cluster      | `gpu_train`            | `gpu_eval`             | `cpu_preprocess` |
|--------------|------------------------|------------------------|------------------|
| **pitzer**   | `gpu`, `gpu:1`         | `gpu`, `gpu:1`         | `cpu`, —         |
| **ascend**   | `nextgen`, `gpu:a100:1`| `nextgen`, `gpu:a100:1`| `nextgen`, —     |
| **cardinal** | `batch`, `gpu:h100:1`  | `batch`, `gpu:h100:1`  | `batch`, —       |

Auto-detected from hostname; overridable via `KD_GAT_CLUSTER`.

#### Training resource profiles

| Model      | Scale | Stage       | Mode            | Time | Mem | CPUs | Workers |
|------------|-------|-------------|-----------------|------|-----|------|---------|
| vgae       | small | autoencoder | gpu_train       | 4:00 | 36G | 4    | 3       |
| vgae       | small | curriculum  | gpu_train       | 2:00 | 36G | 4    | 3       |
| vgae       | large | autoencoder | gpu_train       | 4:00 | 48G | 4    | 3       |
| vgae       | large | curriculum  | gpu_train       | 4:00 | 48G | 4    | 3       |
| gat        | small | normal      | gpu_train       | 4:30 | 36G | 4    | 3       |
| gat        | small | curriculum  | gpu_train       | 4:30 | 36G | 4    | 3       |
| gat        | large | normal      | gpu_train       | 5:00 | 36G | 4    | 3       |
| gat        | large | curriculum  | gpu_train       | 5:00 | 36G | 4    | 3       |
| dgi        | small | autoencoder | gpu_train       | 2:00 | 36G | 4    | 3       |
| dqn        | small | fusion      | gpu_train       | 1:00 | 16G | 2    | 0       |
| dqn        | large | fusion      | gpu_train       | 2:00 | 24G | 2    | 0       |
| preprocess | any   | preprocess  | cpu_preprocess  | 2:00 | 72G | 8    | 0       |
| test       | any   | test        | cpu_preprocess  | 0:30 | 16G | 8    | 0       |

#### Dead / unreachable resource profiles

| Model      | Scale           | Issue                                                         |
|------------|-----------------|---------------------------------------------------------------|
| vgae/gat/dgi/dqn | medium   | `medium` not in `pipeline.yaml` scales                        |
| **bandit** | small/med/large  | Not in `pipeline.yaml`; `get_resources` receives `model_type="dqn"` for all fusion |
| dgi        | large            | Declared in `pipeline.yaml` but missing from `resources.yaml` |

#### Submit profiles (`scripts/submit.sh`)

| Profile            | Partition | CPUs | Mem   | Time  | Mode    | Command                                     |
|--------------------|-----------|------|-------|-------|---------|---------------------------------------------|
| tests              | cpu       | 8    | 16G   | 1:00  | cpu     | `python -m pytest`                          |
| rebuild-caches     | cpu       | 4    | 128G  | 4:00  | cpu-raw | `python -m graphids rebuild-caches`         |
| validate           | cpu       | 4    | 8G    | 0:15  | cpu     | `python -m graphids.orchestrate validate`   |
| ablation           | cpu       | 4    | 8G    | 24:00 | cpu     | `dg launch`                                 |
| profile            | gpudebug  | 4    | 36G   | 1:00  | gpu     | `python -m graphids profile`                |
| probe-budget       | gpudebug  | 8    | 36G   | 1:00  | gpu     | `python -m graphids probe-budget`           |
| landscape          | gpu       | 4    | 32G   | 2:00  | gpu     | `python -m graphids analyze landscape`      |

Modes: `cpu` = `SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1`, `cpu-raw` = `SKIP_CUDA_CONF=1 STAGE_DATA_ARGS=--raw`, `gpu` = full preamble.

#### Adaptive retry (orchestrator)

| Failure        | Scaling      | Max retries |
|----------------|--------------|-------------|
| `OUT_OF_MEMORY`| mem × 1.4    | 2           |
| `TIMEOUT`      | time × 1.5   | 1           |
| `NODE_FAIL`    | (no scaling) | 2           |

---

## 5. IO Configs

### Storage tiers (`write_paths.yaml`)

| Tier            | Path                          | Speed   | Persistence   | Use                               |
|-----------------|-------------------------------|---------|---------------|-----------------------------------|
| **NFS** (home)  | `~/KD-GAT/data/`              | Slow    | Permanent     | Raw data source of truth          |
| **ESS** (GPFS)  | `/fs/ess/PAS1266/kd-gat/`     | Fast    | Permanent     | Lake root: runs, catalog, SLURM logs |
| **Scratch** (GPFS) | `/fs/scratch/PAS1266/`     | Fast    | 90-day purge  | wandb, dagster, data staging      |
| **TMPDIR** (local SSD) | `$TMPDIR/kd-gat-data/` | Fastest | Per-job only  | Training I/O                      |

### Run directory template

```
{lake_root}/dev/{user}/{dataset}/{model}_{scale}_{stage}{identity}{kd_tag}/seed_{seed}
```

Example: `/fs/ess/PAS1266/kd-gat/dev/rf15/set_01/vgae_small_autoencoder_a1b2c3d4/seed_42/`

### Write roles (relative to `run_dir`)

| Component          | Subpath                                      | Source                      |
|--------------------|----------------------------------------------|-----------------------------|
| Best checkpoint    | `checkpoints/best_model.ckpt`                | `CKPT_SUBPATH`              |
| Last checkpoint    | `checkpoints/last.ckpt`                      | `LAST_CKPT_SUBPATH`         |
| CSV metrics        | `lightning_logs/version_*/metrics.csv`       | CSVLogger                   |
| Hparams snapshot   | `lightning_logs/version_*/hparams.yaml`      | `save_hyperparameters()`    |
| Config snapshot    | `lightning_logs/version_*/config.yaml`       | `WandbSaveConfigCallback`   |
| Complete marker    | `.complete`                                  | dagster `_train` (line 424) |

### External write locations

| Component           | Absolute path                                        | Source                     |
|---------------------|------------------------------------------------------|----------------------------|
| Wandb run data      | `/fs/scratch/PAS1266/wandb/`                         | `WANDB_WRITE_DIR`          |
| Dagster IO sidecars | `{lake_root}/.dagster/io/{asset_key}/{partition}.json` | `CheckpointPathIOManager` |
| Dagster event log   | `/fs/scratch/PAS1266/dagster/`                       | `DAGSTER_HOME` (SQLite)    |
| SLURM logs          | `/fs/ess/PAS1266/kd-gat/slurm_logs/{job_name}_%j.{out,err}` | `SLURM_LOG_DIR`  |
| Preprocessing cache | `{lake_root}/cache/v{preprocessing_version}/{dataset}/` | `cache_dir()`           |
| Staging scratch     | `/fs/scratch/PAS1266/kd-gat-data/`                  | `write_paths.yaml:staging.scratch` |
| Staging TMPDIR      | `$TMPDIR/kd-gat-data/`                               | `write_paths.yaml:staging.tmpdir` |
| DuckDB catalog      | `{lake_root}/catalog/kd_gat.duckdb`                  | disposable, rebuildable    |

### Data staging protocol

`_preamble.sh` → `python -m graphids stage-data`:

```
NFS (~/KD-GAT/data/) → Scratch (/fs/scratch/.../kd-gat-data/) → TMPDIR ($TMPDIR/kd-gat-data/)
```

`.staged_marker` skips redundant copies. Scratch purge (90 days) deletes marker → fresh sync.

### Loggers (`trainer.yaml`)

| Logger        | Config                                    | Write location       |
|---------------|-------------------------------------------|----------------------|
| WandbLogger   | `project="kd-gat"`, `log_model=false`     | `WANDB_WRITE_DIR`    |
| CSVLogger     | `save_dir=.` (patched to `default_root_dir` by `cli.py`) | `{run_dir}/lightning_logs/` |

### Training artifacts (all models via Lightning)

| Artifact        | Path (relative to run_dir)                  | Source                     |
|-----------------|---------------------------------------------|----------------------------|
| Best checkpoint | `checkpoints/best_model.ckpt`               | `ModelCheckpoint`          |
| Last checkpoint | `checkpoints/last.ckpt`                     | `ModelCheckpoint(save_last=True)` |
| Hparams snapshot | `lightning_logs/version_*/hparams.yaml`    | `save_hyperparameters()`   |
| Metrics CSV     | `lightning_logs/version_*/metrics.csv`      | CSVLogger                  |
| Config snapshot | `lightning_logs/version_*/config.yaml`      | `WandbSaveConfigCallback`  |
| Complete marker | `.complete`                                 | dagster `_train` after COMPLETED |

VGAE/DGI embed `test_threshold` inside `.ckpt` via `on_save_checkpoint` — not a separate file.

### Logged metrics per model

| Model       | train step                                                        | val step              | test epoch end                                              |
|-------------|-------------------------------------------------------------------|-----------------------|-------------------------------------------------------------|
| VGAE        | `train_loss`                                                      | `val_loss`            | accuracy, f1, precision, recall, specificity, auc, threshold |
| DGI         | `train_loss`                                                      | `val_loss`            | accuracy, f1, precision, recall, specificity, auc, threshold |
| GAT         | `train_loss`, `train_acc`                                         | `val_loss`, `val_acc` | accuracy, f1, precision, recall, specificity, auc           |
| Temporal    | `train_loss`, `train_acc`                                         | `val_loss`, `val_acc` | accuracy, f1, precision, recall, specificity, auc           |
| DQN         | `avg_reward`, `avg_alpha`, `epsilon`, `loss`                      | `val_acc`*            | —                                                           |
| Bandit      | `accuracy`, `avg_reward`, `avg_alpha`, `alpha_std`, `avg_ucb_width`, `backbone_loss` | `val_acc`* | —                                              |
| MLP         | `train_loss`                                                      | `val_loss`, `val_acc` | accuracy, f1, precision, recall, specificity, auc           |
| WeightedAvg | `train_loss`, `alpha`                                             | `val_loss`, `val_acc` | accuracy, f1, precision, recall, specificity, auc           |

*DQN and Bandit `val_acc` comes from shared `FusionModuleBase.validation_step`.

### Post-training analyzer artifacts (`core/artifacts/analyzer.py`)

| Artifact       | File                                  | Contents                                                    |
|----------------|---------------------------------------|-------------------------------------------------------------|
| Embeddings     | `embeddings.npz`                      | `embeddings` [N, latent_dim], `labels` [N], `model_type`   |
| Attention      | `attention_weights.npz`               | `sample_{i}_layer_{j}_alpha` per sample/layer (gatv1 only) |
| CKA            | `cka.json`                            | `{"layer_0": float, ...}` per-layer student/teacher similarity |
| Landscape      | `loss_landscape_{model_type}.parquet` | `x`, `y`, `loss`, `model_type`, `dataset`                  |
| Fusion policy  | `dqn_policy.json`                     | `alphas`, `labels`, `alpha_by_label`, `q_values`           |

#### Analyzer config per model type

| YAML                  | model_type | embeddings | attention | cka | landscape  | fusion_policy |
|-----------------------|-----------|------------|-----------|-----|------------|---------------|
| `analyze_vgae.yaml`   | vgae      | yes        | —         | —   | yes (51×51)| —             |
| `analyze_gat.yaml`    | gat       | yes        | yes       | yes | yes        | —             |
| `analyze_fusion.yaml` | fusion    | —          | —         | —   | —          | yes           |

---

## 6. Orchestration Configs

### Dagster (`component.py`)

#### `SlurmTrainingComponent` fields

| Field            | Type   | Default | Purpose                              |
|------------------|--------|---------|--------------------------------------|
| `dry_run`        | `bool` | `false` | Skip sbatch, log command only        |
| `poll_interval`  | `int`  | `60`    | Seconds between sacct polls          |
| `max_concurrent` | `int`  | `0`     | 0 = unlimited (SLURM handles throttling) |

#### Partitions (`MultiPartitionsDefinition`)

| Dimension | Source          | Values                                 |
|-----------|-----------------|----------------------------------------|
| `dataset` | `datasets.yaml` | hcrl_ch, hcrl_sa, set_01–set_04 (6)   |
| `seed`    | `recipe.sweep.seeds` | [42] (default)                    |

#### Checkpoint handoff (IOManager)

Path: `{lake_root}/.dagster/io/{asset_key}/{partition}.json` → `{"checkpoint_path": "...best_model.ckpt"}`

| Upstream model | CLI flag                           |
|----------------|------------------------------------|
| `vgae`         | `--data.init_args.vgae_ckpt_path`  |
| `dgi`          | `--data.init_args.vgae_ckpt_path`  |
| `gat`          | `--data.init_args.gat_ckpt_path`   |

#### Completion protocol

1. SLURM job writes `best_model.ckpt`
2. `_train` verifies `state == "COMPLETED"`
3. `_train` writes `.complete` marker
4. Returns checkpoint path → IOManager persists to sidecar

Skip logic: if both `best_model.ckpt` + `.complete` exist → idempotent rematerialization.

### SLURM job lifecycle

#### `submit.sh` (login node)
```
submit.sh <profile> [args...]
  → reads resource profile from resources.yaml
  → sources .env for account
  → sbatch --wrap="${PREAMBLE} && ${COMMAND} $*"
```

#### `_preamble.sh` (job node setup)
1. `module load python/3.12`
2. `source .venv/bin/activate`
3. `source .env` (exports KD_GAT_* vars)
4. `umask 002`
5. Set WANDB_* vars
6. Set `PYTORCH_CUDA_ALLOC_CONF` (unless `SKIP_CUDA_CONF=1`)
7. `python -m graphids stage-data` (unless `SKIP_STAGE_DATA=1`)
8. Set `KD_GAT_STAGE_DIR=$TMPDIR/kd-gat-stage`

#### `_epilog.sh` (cleanup)
1. `sacct` accounting summary (Elapsed, MaxRSS, MaxVMSize)
2. Rotate SLURM logs older than 30 days

Signal handling: `--signal=B:USR1@300` → USR1 five minutes before wall time → Lightning `SLURMEnvironment` graceful checkpoint + requeue.

---

## 7. Permutation / Experiment Scale

### Declared dimensions (`pipeline.yaml` + `datasets.yaml`)

| Dimension           | Values                                     | Count |
|---------------------|--------------------------------------------|-------|
| Datasets            | hcrl_ch, hcrl_sa, set_01–04                | 6     |
| Scales              | small, large                               | 2     |
| Autoencoder models  | vgae, dgi                                  | 2     |
| AE variants         | variational=true/false (GAE)               | 2     |
| GAT stages          | normal, curriculum                         | 2     |
| GAT conv types      | gatv2, gat, gps                            | 3     |
| Loss functions      | ce, focal, weighted_ce                     | 3     |
| Fusion methods      | bandit, dqn, mlp, weighted_avg             | 4     |
| Seeds               | 1 (ablation) / 3+ (final eval)             | 1–3   |

**Full cross-product:** 6 × 2 × 2 × 2 × 2 × 3 × 3 × 4 × 1 = **3,456 runs**

### Ablation recipe (`ablation.yaml`) — 18 configs

| Claim                      | Configs | Dimensions varied         |
|----------------------------|---------|---------------------------|
| Loss × Curriculum factorial | 6      | loss_fn(3) × gat_stage(2) |
| Fusion method              | 4       | fusion_method(4)          |
| Conv type                  | 3       | conv_type(3)              |
| Unsupervised method        | 3       | model_type(3: vgae, gae, dgi) |
| Single-model baselines     | 2       | stages subset             |
| KD & scale                 | 2       | scale(2)                  |

With stage dedup → ~36 SLURM jobs (vs 54 naive).

### Scaling projections

| Scenario                           | Jobs   |
|------------------------------------|--------|
| Current ablation (1 dataset, 1 seed) | ~36  |
| All 6 datasets, 1 seed             | ~216   |
| All 6 datasets, 3 seeds            | ~648   |
| Full cross-product, 1 dataset      | ~3,456 |

---

## 8. Known Config Smells

| Issue | Location | Risk |
|-------|----------|------|
| `CurriculumDataModule.max_epochs` must match `trainer.max_epochs` | `curriculum.py:106` vs `trainer.yaml` | Silent wrong curriculum schedule if they drift |
| `FusionDataModule.batch_size` silently ignored for RL methods | `datamodule.py:347` | Misleading config |
| `N_EDGE_FEATURES` comment says 10, actual value is 11 | `features.py:64` | Stale comment |
| Temporal model has no stage YAML | `config/stages/` | Can't launch via `--config` pattern |
| `dgi/large` declared in `pipeline.yaml` but no model config or resource profile | `pipeline.yaml` + `models/dgi/` + `resources.yaml` | Runtime `KeyError` |
| `bandit` resource profiles never reached | `resources.yaml` | `get_resources` receives `model_type="dqn"` for all fusion |
| `medium` scale entries in `resources.yaml` have no `pipeline.yaml` counterpart | `resources.yaml` | Dead config |
| `batch_size=8192` is dead config | `datamodule.py` | Bypassed by `DynamicBatchSampler` when `dynamic_batching=True` |
| `num_workers` mismatch | `resources.yaml` (3) vs stage YAML (2) | 1 CPU wasted per job |
| Temporal `compile_model` unwired | `temporal.py` | Param in `__init__` but no `torch.compile` call |
| MLP/WeightedAvg forced to `precision=32` | `fusion.yaml` | Could run at `16-mixed`, inherits override unnecessarily |
| No LR scheduler for fusion methods | all fusion models | Constant LR, no warmup/decay |
| No `analyze_dgi.yaml` | `config/analyze/` | DGI analyzer would work with `model_type=vgae` but no config |
| No `analyze_temporal.yaml` | `config/analyze/` | No analyzer support for temporal model |
| Landscape skips fusion | `_LOSS_FN` dict | Only has `vgae` and `gat` keys; fusion logs warning, writes nothing |
