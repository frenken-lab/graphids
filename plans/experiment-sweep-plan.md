# Experiment Plan: Ablation Study + Hyperparameter Sweeps

> Status: **partially superseded** | Original: 2026-03-23 | Updated: 2026-03-28
>
> **What's still valid:** Paper claims, ablation axes, config table, stage sharing DAG,
> phase structure (ablation -> HPO -> final eval).
>
> **What's superseded:** Infrastructure sections reference Hydra, `--multirun`, `config.yaml`,
> `models.yaml`, `pipeline/stages/`, `resolve()` -- all deleted. The CLI is now LightningCLI
> with jsonargparse + YAML stages. Orchestration is `graphids/orchestrate/`.

## Context

KD-GAT paper needs experimental evidence for 6 claims. The deployment target is a **small model** on an onboard system; the large model is the teacher/helper. Two experiment types:
1. **Ablation study** -- prove each component matters (factorial + one-at-a-time)
2. **HPO sweep** -- optimize the winning config (Optuna TPE, warm-started from prior sweeps)

**Prior sweep results** at `/fs/ess/PAS1266/kd-gat/sweeps/`:
- `hcrl_ch/`: autoencoder_large, curriculum_large, fusion_large
- `set_01/`: autoencoder_large, autoencoder_small
- Will warm-start new Optuna studies via `study.enqueue_trial()`

**Infrastructure**: ~~Hydra submitit launcher + Optuna sweeper~~ LightningCLI + jsonargparse YAML configs. Orchestration via `graphids/orchestrate/`.

---

## Paper Claims -> Ablation Axes

| # | Claim | Variants |
|---|-------|----------|
| 1 | Fusion > single models | fused vs VGAE-only vs GAT-only |
| 2 | Bandit > other fusion | bandit vs dqn vs mlp vs weighted_avg |
| 3 | KD improves small model | small_kd vs small (large = reference) |
| 4 | Focal + curriculum > alternatives | {focal, ce, weighted_ce} x {curriculum, normal} (2x3 factorial) |
| 5 | GATv2 > other conv layers | GATv2 vs GATv1 vs GPSConv |
| 6 | VGAE > other unsupervised | VGAE vs GAE vs DGI |

---

## Decisions

| Decision | Choice |
|----------|--------|
| Ablation datasets | `hcrl_ch` + `set_01` (2 of 6) |
| Final eval datasets | All 6 |
| Seed strategy | Screen with 1 seed -> expand to 3 for final table |
| LR fairness | Shared HPs from prior sweeps |
| Warm-start | Yes -- enqueue old best trials into new Optuna studies |
| Baseline (vanilla) | small / ce / normal / weighted_avg -- simplest small model |
| Proposed method | small_kd / focal / curriculum / bandit -- full pipeline |
| Large reference | large / focal / curriculum / bandit -- teacher upper bound |

---

## Phase 1: Ablation Study

### 1a. Screening Pass (1 seed, 2 datasets)

**Paper narrative**: Start from vanilla small model, add components, show each helps.

#### Ablation configs

| # | Name | Scale | Conv | Unsup | Loss | GAT stage | Fusion | Tests claim |
|---|------|-------|------|-------|------|-----------|--------|-------------|
| **Floor** |
| 1 | Vanilla | small | gatv2 | VGAE | ce | normal | weighted_avg | Floor |
| **Loss x Curriculum factorial (claim 4)** |
| 2 | +focal | small | gatv2 | VGAE | focal | normal | weighted_avg | focal alone |
| 3 | +curriculum | small | gatv2 | VGAE | ce | curriculum | weighted_avg | curriculum alone |
| 4 | +focal+curriculum | small | gatv2 | VGAE | focal | curriculum | weighted_avg | interaction |
| 5 | wce+normal | small | gatv2 | VGAE | weighted_ce | normal | weighted_avg | wce alone |
| 6 | wce+curriculum | small | gatv2 | VGAE | weighted_ce | curriculum | weighted_avg | wce+curriculum |
| **Fusion method (claim 2)** -- all share config 4 upstream |
| 7 | +bandit | small | gatv2 | VGAE | focal | curriculum | bandit | best fusion? |
| 8 | +dqn | small | gatv2 | VGAE | focal | curriculum | dqn | |
| 9 | +mlp | small | gatv2 | VGAE | focal | curriculum | mlp | |
| **KD & scale (claim 3)** |
| 10 | +KD (proposed) | small_kd | gatv2 | VGAE | focal | curriculum | bandit | KD helps? |
| 11 | Large (teacher ref) | large | gatv2 | VGAE | focal | curriculum | bandit | upper bound |
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

Many configs share upstream stages. The pipeline is `Autoencoder -> GAT -> Fusion -> Eval`.

**Unique autoencoder runs (7 per dataset):**

| ID | Scale | Conv | Method | Used by configs |
|----|-------|------|--------|-----------------|
| V1 | small | gatv2 | VGAE | 1-10, 16, 17 |
| V2 | small | gatv2 | GAE | 14 |
| V3 | small | gatv2 | DGI | 15 |
| V4 | small | gat | VGAE | 12 |
| V5 | small | gps | VGAE | 13 |
| V6 | large | gatv2 | VGAE | 11 (+ teacher for V7) |
| V7 | small_kd | gatv2 | VGAE | 10 (needs V6 as teacher) |

**Unique GAT runs (12 per dataset):**

| ID | Upstream | Conv | Loss | Stage | Used by configs |
|----|----------|------|------|-------|-----------------|
| G1 | V1 | gatv2 | focal | curriculum | 4, 7, 8, 9 (fusion comparison) |
| G2 | V1 | gatv2 | ce | normal | 1 (vanilla) |
| G3 | V1 | gatv2 | focal | normal | 2, 17 (GAT-only) |
| G4 | V1 | gatv2 | ce | curriculum | 3 |
| G5 | V1 | gatv2 | wce | normal | 5 |
| G6 | V1 | gatv2 | wce | curriculum | 6 |
| G7 | V2 | gatv2 | focal | curriculum | 14 (GAE upstream) |
| G8 | V3 | gatv2 | focal | curriculum | 15 (DGI upstream) |
| G9 | V4 | gat | focal | curriculum | 12 |
| G10 | V5 | gps | focal | curriculum | 13 |
| G11 | V6 | gatv2 | focal | curriculum | 11 (large) |
| G12 | V7 | gatv2 | focal | curriculum | 10 (KD student) |

### Job counts (per dataset per seed)

| Stage | Unique runs | GPU jobs (with batching) |
|-------|-------------|------------------------|
| Autoencoder | 7 | 7 |
| GAT | 12 | 12 |
| Fusion+Eval | 15 fused + 2 single-model | ~6 batched |
| **Total** | | **~25 GPU jobs** |

**Screening: 25 jobs x 2 datasets = ~50 GPU jobs**

### 1b. Full Ablation (3 seeds, 2 datasets)

After screening identifies clear winners: run 3 seeds for configs in the final paper table.

**Max**: 17 x 2 x 3 = 102 pipelines, ~150 GPU jobs (with sharing)

---

## Phase 2: HPO on Winning Config

After ablation locks the component combination:

1. **Warm-start** Optuna with prior trials from `/fs/ess/PAS1266/kd-gat/sweeps/`
2. **Per-stage search spaces**:

| Stage | Parameters to tune |
|-------|-------------------|
| Autoencoder | lr, weight_decay, latent_dim, dropout, heads, embedding_dim, proj_dim |
| GAT | lr, weight_decay, hidden, layers, heads, dropout, fc_layers, proj_dim, focal_gamma |
| Fusion | lr, hidden, layers, method-specific params |

3. **Budget**: 30 trials x stages x datasets

---

## Phase 3: Final Evaluation

- Best config from Phase 2
- All 6 datasets x 3+ seeds
- Metrics: F1, AP, AUC (mean +/- std)
- Statistical tests (paired t-test or Wilcoxon)

---

## Implementation: Current CLI

### How ablation configs are run now

Each ablation config is a YAML file in `graphids/config/stages/` + optional overlay:

```bash
# Training
python -m graphids fit --config graphids/config/stages/autoencoder.yaml
python -m graphids fit --config graphids/config/stages/normal.yaml \
                       --config graphids/config/overlays/small_gat.yaml

# Analysis artifacts
python -m graphids analyze --config graphids/config/stages/analyze_vgae.yaml \
    --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa
```

### Orchestration

Dagster-based. See `plans/architecture/dagster-native-orchestration.md` for current design.

```bash
# Submission
sbatch scripts/slurm/run_ablation.sh
# Or directly: python -m graphids.orchestrate --partition "set_01|42"
```

### Key files

| File | Role |
|------|------|
| `graphids/config/pipeline.yaml` | Stage DAG, identity_keys, valid models/scales |
| `graphids/config/ablation.yaml` | 18-config ablation recipe |
| `graphids/config/stages/*.yaml` | Per-stage LightningCLI configs |
| `graphids/config/overlays/*.yaml` | Scale/ablation variant overlays |
| `graphids/config/resources.yaml` | SLURM resource profiles |
| `graphids/orchestrate/dagster_defs.py` | Dagster asset definitions |
| `/fs/ess/PAS1266/kd-gat/sweeps/` | Prior sweep results |

---

## Superseded sections

<details>
<summary>Original infrastructure design (Hydra-based, 2026-03-23)</summary>

The original plan used:
- `graphids/config/config.yaml` (Hydra root config) -- **deleted**
- `graphids/config/models.yaml` (model x scale presets) -- **deleted**
- `graphids/config/__init__.py` with `TrainingConfig`, `PipelineConfig` -- **replaced by flat YAML**
- `graphids/pipeline/stages/modules.py` (loss dispatch, DataLoader setup) -- **deleted**
- `graphids/pipeline/stages/fusion.py` (fusion method dispatch) -- **deleted**
- Hydra submitit launcher + Optuna sweeper via `--multirun` -- **replaced by LightningCLI**
- `graphids.config.resolve()` for config validation -- **deleted**
- Python orchestrator using `resolve()` at `scripts/slurm/orchestrate_ablation.py` -- **replaced by `graphids/orchestrate/`**

HPO via Optuna is still planned but will integrate differently (likely Optuna + LightningCLI
rather than Hydra sweeper).

</details>

---

## Prerequisites status

| Change | Status |
|--------|--------|
| GAE unsupervised method | Done (autoencoder.yaml flag) |
| DGI unsupervised method | Done (DGIModule + config) |
| VGAE-only evaluation path | Done (LightningCLI test) |
| GAT-only evaluation path | Done (LightningCLI test) |
| Cache rebuild (6/6 datasets) | Done |
| Config system (Hydra -> jsonargparse) | Done |
