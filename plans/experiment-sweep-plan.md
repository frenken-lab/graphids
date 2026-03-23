# Experiment Plan: Ablation Study + Hyperparameter Sweeps

## Context

KD-GAT paper needs experimental evidence for 5 claims about the pipeline. Two experiment types:
1. **Ablation study** — prove each component matters (discrete A/B, one-at-a-time)
2. **HPO sweep** — optimize the winning config (Optuna TPE, warm-started from prior sweeps)

**Prior sweep results** at `/fs/ess/PAS1266/kd-gat/sweeps/`:
- `hcrl_ch/`: autoencoder_large, curriculum_large, fusion_large (complete)
- `set_01/`: autoencoder_large, autoencoder_small
- Will warm-start new Optuna studies with these as seed trials (`study.enqueue_trial()`)

**Infrastructure**: Hydra submitit launcher + Optuna sweeper configured in `config.yaml:43-69`.

---

## Paper Claims → Ablation Axes

| # | Claim | Ablation |
|---|-------|----------|
| 1 | Fusion > single models | Fused vs VGAE-only vs GAT-only |
| 2 | Bandit > other fusion methods | bandit vs dqn vs mlp vs weighted_avg |
| 3 | KD improves small model | small_kd vs small_nokd (large = reference) |
| 4 | Focal+curriculum > alternatives | focal vs ce vs weighted_ce; curriculum vs normal stage |
| 5 | GAT > other GNN architectures | **Deferred** (~20 LOC per alternative via conv_type swap) |

---

## Decisions

| Decision | Choice |
|----------|--------|
| Ablation datasets | `hcrl_ch` + `set_01` (2 of 6) |
| Final eval datasets | All 6 |
| Seed strategy | Screen with 1 seed → expand to 3 for final table |
| LR fairness | Shared HPs from prior sweeps |
| Warm-start | Yes — enqueue old best trials into new Optuna studies |
| Baseline | large / focal / curriculum / bandit (proposed method) |
| GAT-alone baseline | `normal` stage (no VGAE dependency, fairer comparison) |
| Claim 5 | Deferred — architectures not yet implemented |

---

## Phase 1: Ablation Study

### 1a. Screening Pass (1 seed, 2 datasets)

**Baseline**: large scale, focal loss, curriculum stage, bandit fusion

| # | Name | Scale | Loss | GAT stage | Fusion | Claim |
|---|------|-------|------|-----------|--------|-------|
| 1 | **Proposed method** | large | focal | curriculum | bandit | — |
| 2 | VGAE-only | large | — | — | — | 1 |
| 3 | GAT-only | large | focal | normal | — | 1 |
| 4 | Fusion: DQN | large | focal | curriculum | dqn | 2 |
| 5 | Fusion: MLP | large | focal | curriculum | mlp | 2 |
| 6 | Fusion: weighted avg | large | focal | curriculum | weighted_avg | 2 |
| 7 | Student + KD | small_kd | focal | curriculum | bandit | 3 |
| 8 | Student no KD | small_nokd | focal | curriculum | bandit | 3 |
| 9 | Loss: CE | large | ce | curriculum | bandit | 4 |
| 10 | Loss: weighted CE | large | weighted_ce | curriculum | bandit | 4 |
| 11 | No curriculum | large | focal | normal | bandit | 4 |

**11 configs × 2 datasets × 1 seed = 22 experiment pipelines**

### Stage sharing (compute reuse)

| Stage | Unique configs per (dataset, seed) | Notes |
|-------|-----------------------------------|-------|
| VGAE | 3 | large, small_kd, small_nokd |
| GAT | 6 | {large,small_kd,small_nokd}×focal×curriculum + large×{ce,wce}×curriculum + large×focal×normal |
| Fusion | 8 | configs 1,4,5,6,7,8,9,10,11 minus configs 2,3 (skip fusion) minus sharing |
| Eval | 11 | every config needs evaluation |

**Screening job count**: ~56 total SLURM jobs (across all stages)

### 1b. Full Ablation (3 seeds, 2 datasets)

After screening identifies clear winners: run 3 seeds for configs that make the final paper table.

**Max**: 11 × 2 × 3 = 66 pipelines, ~174 jobs

---

## Phase 2: HPO on Winning Config

After ablation locks the component combination:

1. **Warm-start** Optuna with prior trials from `/fs/ess/PAS1266/kd-gat/sweeps/`
2. **Sweeper**: Optuna TPE, 30 trials
3. **Launcher**: submitit SLURM (`config.yaml:60-69`)
4. **Per-stage search spaces**:

| Stage | Parameters to tune |
|-------|-------------------|
| VGAE | lr, weight_decay, latent_dim, dropout, heads, embedding_dim, proj_dim |
| GAT | lr, weight_decay, hidden, layers, heads, dropout, fc_layers, proj_dim, focal_gamma |
| Fusion | lr, hidden, layers, method-specific (gamma, epsilon for DQN; UCB params for bandit) |

5. **Budget**: 30 trials × stages × datasets

---

## Phase 3: Final Evaluation

- Best config from Phase 2
- All 6 datasets × 3+ seeds
- Metrics: F1, AP, AUC (mean ± std)
- Statistical tests (paired t-test or Wilcoxon)

---

## Implementation Steps

### Step 0: Verify blockers clear
- [x] Cache rebuild submitted (job 45971172)
- [x] Tests submitted (job 45971174)
- [x] Smoke test submitted (job 45971179, depends on cache)

### Step 1: Ablation orchestrator
Create a script/config that:
- Generates the 11 CLI invocations per (dataset, seed)
- Handles stage dependencies (VGAE → GAT → Fusion → Eval)
- Reuses shared upstream stages (e.g., one large VGAE serves configs 1, 3-6, 9-11)
- Submits via sbatch with `--dependency=afterok:$VGAE_JOB`

**Decision**: Individual sbatch per stage with dependency chaining vs Hydra submitit `--multirun`.
- Submitit `--multirun` works well for sweeping ONE stage.
- For multi-stage pipelines with sharing, a shell orchestrator with sbatch dependencies is more natural.
- **Recommendation**: Shell orchestrator that calls `python -m graphids` per stage, wired with SLURM dependencies.

### Step 2: Warm-start utility
- Read YAML files from `/fs/ess/PAS1266/kd-gat/sweeps/`
- Create `optuna.Study`, call `study.enqueue_trial(params={...})`
- Feed into Hydra-Optuna sweeper

### Step 3: Single-model baselines (configs 2, 3)
- **VGAE-only** (config 2): verify evaluation stage can score using only VGAE anomaly scores
- **GAT-only** (config 3): verify evaluation stage can score using only GAT predictions from `normal` stage
- May need small evaluation stage changes to support single-model inputs

### Step 4: Run screening (1 seed)
### Step 5: Analyze, decide which configs get 3-seed treatment
### Step 6: Run full ablation (3 seeds)
### Step 7: HPO on winner
### Step 8: Final evaluation on all 6 datasets

---

## Key Files

| File | Role |
|------|------|
| `graphids/config/config.yaml:43-69` | Hydra launcher + sweeper config |
| `graphids/config/pipeline.yaml` | Stage DAG, variants (large/small_kd/small_nokd) |
| `graphids/config/models.yaml` | Model × scale presets |
| `graphids/config/__init__.py:119-156` | TrainingConfig (loss_fn, curriculum params, etc.) |
| `graphids/pipeline/stages/modules.py:295-304` | Loss function dispatch (ce/weighted_ce/focal) |
| `graphids/pipeline/stages/fusion.py:331-342` | Fusion method dispatch (dqn/bandit/mlp/weighted_avg) |
| `/fs/ess/PAS1266/kd-gat/sweeps/` | Prior sweep results for warm-starting |
| `scripts/slurm/_preamble.sh` | Shared SLURM job setup |
