# Experiment Plan: Ablation Study + Hyperparameter Sweeps

> Status: **active** (experiment design) | Created: 2026-03-23 | Audited: 2026-03-30
>
> Infrastructure uses LightningCLI + jsonargparse + dagster orchestration.
> Hydra/submitit references removed (superseded 2026-03-26).

## Paper Claims -> Ablation Axes

| # | Claim | Variants |
|---|-------|----------|
| 1 | Fusion > single models | fused vs VGAE-only vs GAT-only |
| 2 | Bandit > other fusion | bandit vs dqn vs mlp vs weighted_avg |
| 3 | KD improves small model | small_kd vs small (large = reference) |
| 4 | Focal + curriculum > alternatives | {focal, ce, weighted_ce} x {curriculum, normal} (2x3) |
| 5 | GATv2 > other conv layers | GATv2 vs GATv1 vs GPSConv |
| 6 | VGAE > other unsupervised | VGAE vs GAE vs DGI |

## Decisions

| Decision | Choice |
|----------|--------|
| Ablation datasets | `hcrl_ch` + `set_01` (2 of 6) |
| Final eval datasets | All 6 |
| Seed strategy | Screen with 1 seed -> expand to 3 for final table |
| LR fairness | Shared HPs from prior sweeps |
| Warm-start | Yes — enqueue old best trials into new Optuna studies |
| Baseline (vanilla) | small / ce / normal / weighted_avg |
| Proposed method | small_kd / focal / curriculum / bandit |
| Large reference | large / focal / curriculum / bandit |

## Phase 1: Ablation Study

### 1a. Screening Pass (1 seed, 2 datasets)

#### 17 ablation configs

| # | Name | Scale | Conv | Unsup | Loss | GAT stage | Fusion | Tests claim |
|---|------|-------|------|-------|------|-----------|--------|-------------|
| **Floor** |
| 1 | Vanilla | small | gatv2 | VGAE | ce | normal | weighted_avg | Floor |
| **Loss x Curriculum (claim 4)** |
| 2 | +focal | small | gatv2 | VGAE | focal | normal | weighted_avg | focal alone |
| 3 | +curriculum | small | gatv2 | VGAE | ce | curriculum | weighted_avg | curriculum alone |
| 4 | +focal+curriculum | small | gatv2 | VGAE | focal | curriculum | weighted_avg | interaction |
| 5 | wce+normal | small | gatv2 | VGAE | weighted_ce | normal | weighted_avg | wce alone |
| 6 | wce+curriculum | small | gatv2 | VGAE | weighted_ce | curriculum | weighted_avg | wce+curriculum |
| **Fusion method (claim 2)** — share config 4 upstream |
| 7 | +bandit | small | gatv2 | VGAE | focal | curriculum | bandit | best fusion? |
| 8 | +dqn | small | gatv2 | VGAE | focal | curriculum | dqn | |
| 9 | +mlp | small | gatv2 | VGAE | focal | curriculum | mlp | |
| **KD & scale (claim 3)** |
| 10 | +KD (proposed) | small_kd | gatv2 | VGAE | focal | curriculum | bandit | KD helps? |
| 11 | Large (teacher) | large | gatv2 | VGAE | focal | curriculum | bandit | upper bound |
| **Conv type (claim 5)** |
| 12 | GATv1 | small | gat | VGAE | focal | curriculum | bandit | GATv2 > v1? |
| 13 | GPSConv | small | gps | VGAE | focal | curriculum | bandit | GATv2 > GPS? |
| **Unsupervised method (claim 6)** |
| 14 | GAE | small | gatv2 | GAE | focal | curriculum | bandit | need variational? |
| 15 | DGI | small | gatv2 | DGI | focal | curriculum | bandit | recon vs contrastive? |
| **Single-model baselines (claim 1)** |
| 16 | VGAE-only | small | gatv2 | VGAE | -- | -- | -- | need classifier? |
| 17 | GAT-only | small | gatv2 | -- | focal | normal | -- | need fusion? |

**17 configs x 2 datasets x 1 seed = 34 screening pipelines**

### Stage sharing DAG

**7 unique autoencoder runs per dataset:**

| ID | Scale | Conv | Method | Used by configs |
|----|-------|------|--------|-----------------|
| V1 | small | gatv2 | VGAE | 1-10, 16, 17 |
| V2 | small | gatv2 | GAE | 14 |
| V3 | small | gatv2 | DGI | 15 |
| V4 | small | gat | VGAE | 12 |
| V5 | small | gps | VGAE | 13 |
| V6 | large | gatv2 | VGAE | 11 (+ teacher for V7) |
| V7 | small_kd | gatv2 | VGAE | 10 (needs V6 as teacher) |

**12 unique GAT runs per dataset:**

| ID | Upstream | Conv | Loss | Stage | Used by configs |
|----|----------|------|------|-------|-----------------|
| G1 | V1 | gatv2 | focal | curriculum | 4, 7, 8, 9 |
| G2 | V1 | gatv2 | ce | normal | 1 |
| G3 | V1 | gatv2 | focal | normal | 2, 17 |
| G4 | V1 | gatv2 | ce | curriculum | 3 |
| G5 | V1 | gatv2 | wce | normal | 5 |
| G6 | V1 | gatv2 | wce | curriculum | 6 |
| G7 | V2 | gatv2 | focal | curriculum | 14 |
| G8 | V3 | gatv2 | focal | curriculum | 15 |
| G9 | V4 | gat | focal | curriculum | 12 |
| G10 | V5 | gps | focal | curriculum | 13 |
| G11 | V6 | gatv2 | focal | curriculum | 11 |
| G12 | V7 | gatv2 | focal | curriculum | 10 |

**Screening: ~25 GPU jobs x 2 datasets = ~50 GPU jobs**

### 1b. Full Ablation (3 seeds, 2 datasets)

After screening: 3 seeds for configs in the final paper table.
Max: 17 x 2 x 3 = 102 pipelines, ~150 GPU jobs (with sharing).

## Phase 2: HPO on Winning Config

1. Warm-start Optuna with prior trials from `/fs/ess/PAS1266/kd-gat/sweeps/`
2. Per-stage search spaces: lr, weight_decay, dims, dropout, heads, method-specific params
3. Budget: 30 trials x stages x datasets

## Phase 3: Final Evaluation

- Best config from Phase 2, all 6 datasets, 3+ seeds
- Metrics: F1, AP, AUC (mean +/- std)
- Statistical tests (paired t-test or Wilcoxon)

## Key files

| File | Role |
|------|------|
| `graphids/config/pipeline.yaml` | Stage DAG, identity_keys, valid models/scales |
| `graphids/config/recipes/ablation.yaml` | 18-config ablation recipe |
| `graphids/config/stages/*.yaml` | Per-stage LightningCLI configs |
| `graphids/config/overlays/*.yaml` | Scale/ablation variant overlays |
| `graphids/config/resources.yaml` | SLURM resource profiles |
| `graphids/orchestrate/definitions.py` | Dagster asset definitions |

## Prerequisites (all complete)

GAE unsupervised method, DGI module, VGAE-only eval path, GAT-only eval path,
cache rebuild (6/6 datasets), config system (jsonargparse).
