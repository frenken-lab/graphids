# Experiment Plan: Ablation Study + Hyperparameter Sweeps

## Context

KD-GAT paper needs experimental evidence for 6 claims. The deployment target is a **small model** on an onboard system; the large model is the teacher/helper. Two experiment types:
1. **Ablation study** — prove each component matters (factorial + one-at-a-time)
2. **HPO sweep** — optimize the winning config (Optuna TPE, warm-started from prior sweeps)

**Prior sweep results** at `/fs/ess/PAS1266/kd-gat/sweeps/`:
- `hcrl_ch/`: autoencoder_large, curriculum_large, fusion_large
- `set_01/`: autoencoder_large, autoencoder_small
- Will warm-start new Optuna studies via `study.enqueue_trial()`

**Infrastructure**: Hydra submitit launcher + Optuna sweeper via CLI overrides with `--multirun`.

---

## Paper Claims → Ablation Axes

| # | Claim | Variants |
|---|-------|----------|
| 1 | Fusion > single models | fused vs VGAE-only vs GAT-only |
| 2 | Bandit > other fusion | bandit vs dqn vs mlp vs weighted_avg |
| 3 | KD improves small model | small_kd vs small (large = reference) |
| 4 | Focal + curriculum > alternatives | {focal, ce, weighted_ce} × {curriculum, normal} (2×3 factorial) |
| 5 | GATv2 > other conv layers | GATv2 vs GATv1 vs GPSConv |
| 6 | VGAE > other unsupervised | VGAE vs GAE vs DGI |

---

## Decisions

| Decision | Choice |
|----------|--------|
| Ablation datasets | `hcrl_ch` + `set_01` (2 of 6) |
| Final eval datasets | All 6 |
| Seed strategy | Screen with 1 seed → expand to 3 for final table |
| LR fairness | Shared HPs from prior sweeps |
| Warm-start | Yes — enqueue old best trials into new Optuna studies |
| Baseline (vanilla) | small / ce / normal / weighted_avg — simplest small model |
| Proposed method | small_kd / focal / curriculum / bandit — full pipeline |
| Large reference | large / focal / curriculum / bandit — teacher upper bound |

---

## Phase 1: Ablation Study

### 1a. Screening Pass (1 seed, 2 datasets)

**Paper narrative**: Start from vanilla small model, add components, show each helps.

#### Ablation configs

| # | Name | Scale | Conv | Unsup | Loss | GAT stage | Fusion | Tests claim |
|---|------|-------|------|-------|------|-----------|--------|-------------|
| **Floor** |
| 1 | Vanilla | small | gatv2 | VGAE | ce | normal | weighted_avg | Floor |
| **Loss × Curriculum factorial (claim 4)** |
| 2 | +focal | small | gatv2 | VGAE | focal | normal | weighted_avg | focal alone |
| 3 | +curriculum | small | gatv2 | VGAE | ce | curriculum | weighted_avg | curriculum alone |
| 4 | +focal+curriculum | small | gatv2 | VGAE | focal | curriculum | weighted_avg | interaction |
| 5 | wce+normal | small | gatv2 | VGAE | weighted_ce | normal | weighted_avg | wce alone |
| 6 | wce+curriculum | small | gatv2 | VGAE | weighted_ce | curriculum | weighted_avg | wce+curriculum |
| **Fusion method (claim 2)** — all share config 4 upstream |
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
| 16 | VGAE-only | small | gatv2 | VGAE | — | — | — | need classifier? |
| 17 | GAT-only | small | gatv2 | — | focal | normal | — | need fusion? |

**17 configs × 2 datasets × 1 seed = 34 screening pipelines**

### Stage sharing DAG

Many configs share upstream stages. The pipeline is `Autoencoder → GAT → Fusion → Eval`.

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

**Fusion runs — batched sequentially in shared GPU jobs:**

| Job | Runs sequentially | Upstream |
|-----|-------------------|----------|
| F-batch-1 | configs 4(w_avg), 7(bandit), 8(dqn), 9(mlp) | G1 |
| F-batch-2 | configs 1(w_avg), 2(w_avg) | G2, G3 |
| F-batch-3 | configs 3, 5, 6 (bandit each) | G4, G5, G6 |
| F-single | configs 10-15 (bandit each) | G7-G12 |
| (skip) | configs 16, 17 — no fusion | |

### Job counts (per dataset per seed)

| Stage | Unique runs | GPU jobs (with batching) |
|-------|-------------|------------------------|
| Autoencoder | 7 | 7 |
| GAT | 12 | 12 |
| Fusion+Eval | 15 fused + 2 single-model | ~6 batched |
| **Total** | | **~25 GPU jobs** |

**Screening: 25 jobs × 2 datasets = ~50 GPU jobs**

### 1b. Full Ablation (3 seeds, 2 datasets)

After screening identifies clear winners: run 3 seeds for configs in the final paper table.

**Max**: 17 × 2 × 3 = 102 pipelines, ~150 GPU jobs (with sharing)

---

## Phase 2: HPO on Winning Config

After ablation locks the component combination:

1. **Warm-start** Optuna with prior trials from `/fs/ess/PAS1266/kd-gat/sweeps/`
2. **Sweeper**: Optuna TPE, 30 trials, via CLI: `--multirun hydra/sweeper=optuna hydra/launcher=submitit_slurm`
3. **Per-stage search spaces**:

| Stage | Parameters to tune |
|-------|-------------------|
| Autoencoder | lr, weight_decay, latent_dim, dropout, heads, embedding_dim, proj_dim |
| GAT | lr, weight_decay, hidden, layers, heads, dropout, fc_layers, proj_dim, focal_gamma |
| Fusion | lr, hidden, layers, method-specific params |

4. **Budget**: 30 trials × stages × datasets

---

## Phase 3: Final Evaluation

- Best config from Phase 2
- All 6 datasets × 3+ seeds
- Metrics: F1, AP, AUC (mean ± std)
- Statistical tests (paired t-test or Wilcoxon)

---

## Implementation: Orchestrator Design

### Architecture: YAML manifest → orchestrator script

```
ablation.yaml          scripts/slurm/orchestrate_ablation.sh
(declarative plan) ──→ (reads YAML, submits jobs with SLURM dependencies)
```

**Why YAML manifest + orchestrator** (not per-config sbatch scripts):
- 17 configs × multiple stages = 60+ sbatch files would be unmaintainable
- YAML is human-readable: easy to add/remove configs, review the plan
- Orchestrator handles DAG dependencies, stage sharing, job batching
- Same orchestrator works for screening (1 seed) and full (3 seeds)

**Why not Hydra submitit `--multirun`**:
- `--multirun` sweeps ONE stage. Our pipeline has 4 stages with dependencies.
- Stage sharing (one VGAE serving many GATs) requires DAG-aware submission.
- submitit is great for Phase 2 (HPO within a single stage), not Phase 1.

### ablation.yaml structure (draft)

```yaml
# ablation.yaml — Declarative experiment plan
# Orchestrator reads this, resolves sharing, submits SLURM jobs.

defaults:
  datasets: [hcrl_ch, set_01]
  seeds: [42]
  scale: small
  conv_type: gatv2
  unsupervised: vgae
  loss_fn: focal
  gat_stage: curriculum
  fusion_method: bandit

# Each config overrides defaults. Orchestrator computes unique
# (autoencoder, GAT, fusion) stages and shares where possible.
configs:
  # --- Floor ---
  vanilla:
    loss_fn: ce
    gat_stage: normal
    fusion_method: weighted_avg

  # --- Loss × Curriculum factorial ---
  focal_normal:
    gat_stage: normal
    fusion_method: weighted_avg
  ce_curriculum:
    loss_fn: ce
    fusion_method: weighted_avg
  focal_curriculum:
    fusion_method: weighted_avg
  wce_normal:
    loss_fn: weighted_ce
    gat_stage: normal
    fusion_method: weighted_avg
  wce_curriculum:
    loss_fn: weighted_ce
    fusion_method: weighted_avg

  # --- Fusion comparison (all share focal+curriculum upstream) ---
  fusion_bandit: {}   # default = bandit
  fusion_dqn:
    fusion_method: dqn
  fusion_mlp:
    fusion_method: mlp

  # --- KD & scale ---
  kd_student:
    scale: small_kd
  large_reference:
    scale: large

  # --- Conv type ---
  conv_gatv1:
    conv_type: gat
  conv_gps:
    conv_type: gps

  # --- Unsupervised method ---
  unsup_gae:
    unsupervised: gae
  unsup_dgi:
    unsupervised: dgi

  # --- Single-model baselines ---
  vgae_only:
    skip_stages: [curriculum, fusion]
  gat_only:
    gat_stage: normal
    skip_stages: [autoencoder, fusion]
```

### Orchestrator logic (pseudocode)

```python
# orchestrate_ablation.py — reads ablation.yaml, submits SLURM DAG

def main(manifest_path, dry_run=False):
    plan = load_yaml(manifest_path)

    # 1. Resolve each config into full (autoencoder, gat, fusion) specs
    configs = resolve_configs(plan)

    # 2. Deduplicate: group configs by shared upstream stages
    unique_ae = deduplicate(configs, key=ae_key)   # 7 unique
    unique_gat = deduplicate(configs, key=gat_key)  # 12 unique
    fusion_batches = group_by_upstream(configs)      # batch same-upstream

    # 3. Submit autoencoder jobs (no dependencies)
    ae_jobs = {}
    for ae_spec in unique_ae:
        for ds in plan.datasets:
            for seed in plan.seeds:
                job_id = sbatch(stage="autoencoder", **ae_spec, ds, seed)
                ae_jobs[(ae_spec.key, ds, seed)] = job_id

    # 4. Submit GAT jobs (depend on their autoencoder)
    gat_jobs = {}
    for gat_spec in unique_gat:
        ae_dep = ae_jobs[(gat_spec.ae_key, ds, seed)]
        job_id = sbatch(stage=gat_spec.stage, dependency=ae_dep, ...)
        gat_jobs[(gat_spec.key, ds, seed)] = job_id

    # 5. Submit fusion+eval batch jobs (depend on their GAT)
    for batch in fusion_batches:
        gat_dep = gat_jobs[(batch.gat_key, ds, seed)]
        # Single job runs multiple fusion methods sequentially
        sbatch(stage="fusion_batch", methods=batch.methods, dependency=gat_dep)

    # 6. Print job summary
    print_dag(ae_jobs, gat_jobs, fusion_batches)
```

### Open question: orchestrator language

| Approach | Pros | Cons |
|----------|------|------|
| **Python script** | Can import `graphids.config` for validation, rich DAG logic, YAML parsing | Another Python entry point to maintain |
| **Bash script** | Matches existing `scripts/slurm/` patterns, simple sbatch calls | DAG dedup logic is painful in bash |
| **Python generates bash** | Python does planning + validation, outputs a runnable `.sh` | Two-step workflow but clean separation |

**Recommendation**: Python script that reads YAML, validates configs, and calls `sbatch` directly via `subprocess`. Lives at `scripts/slurm/orchestrate_ablation.py`. Uses `graphids.config.resolve()` to validate each config before submission.

---

## Prerequisites (before orchestrator)

### Code changes needed

| Change | Effort | File |
|--------|--------|------|
| Add GCN + SAGE conv types | ~4 lines | `graphids/core/models/_utils.py:82` |
| Add GAE unsupervised method | ~20 lines | New option in autoencoder stage |
| Add DGI unsupervised method | ~50 lines | Different training loop |
| VGAE-only evaluation path | Verify | `graphids/pipeline/stages/evaluation.py` |
| GAT-only evaluation path | Verify | `graphids/pipeline/stages/evaluation.py` |

### Blockers cleared

- [x] Cache rebuild (6/6 datasets with cache_metadata.json)
- [x] Tests (96/96 passed)
- [x] Smoke test (2 epochs, num_workers=0, torch.compile, DynamicBatchSampler)
- [x] Preprocessing parallelism fixed (fork + numpy IPC, 3.3x speedup)
- [x] Training IPC eliminated (num_workers=0, pin_memory=True)
- [x] Hydra config fixed (sweeper/launcher removed from defaults)

---

## Key Files

| File | Role |
|------|------|
| `graphids/config/config.yaml` | Hydra root config |
| `graphids/config/pipeline.yaml` | Stage DAG, variants |
| `graphids/config/models.yaml` | Model × scale presets |
| `graphids/config/__init__.py` | TrainingConfig, PipelineConfig |
| `graphids/core/models/_utils.py:82` | `_make_conv` factory (conv_type dispatch) |
| `graphids/pipeline/stages/modules.py` | Loss dispatch, DataLoader setup |
| `graphids/pipeline/stages/fusion.py` | Fusion method dispatch |
| `/fs/ess/PAS1266/kd-gat/sweeps/` | Prior sweep results |
| `scripts/slurm/_preamble.sh` | Shared SLURM job setup |
