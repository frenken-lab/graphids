# KD-GAT Session Plan

> Last updated: 2026-03-25

## Active Plan

### Ablation Run 003 — Submitting now

18 configs × 2 datasets (set_01, set_02) × 1 seed (42), deduped to 62 SLURM jobs. KD configs deferred.

**Fixes applied (Run 001 + Run 002 post-mortems + 2026-03-25 hardening):**
- [x] `batch_size` default: 4096 → 8192 (VRAM was 33% utilized on V100)
- [x] **VRAM-aware budget**: `compute_node_budget()` auto-caps GPS O(N²) at ~20K nodes from `torch.cuda` VRAM. Replaces manual `batch_size=256` band-aid.
- [x] GPU wall time: 120 → 240 min (7 jobs timed out at ~1:50)
- [x] Eval/fusion wall time: 30 → 60 min
- [x] Dataset-scoped staging: `--dataset` flag copies ~5 GB instead of 87 GB
- [x] CPU eval skips TMPDIR: `--skip-tmpdir` reads from scratch (0 copy)
- [x] Stale cache cleanup: removed v3/v4/v5/v7 dirs (64 GB freed)
- [x] **Identity-aware paths**: `identity_hash` OmegaConf resolver prevents run dir collisions
- [x] **metrics.json**: evaluation stage now persists metrics to disk
- [x] **DuckDB catalog**: `_append_to_catalog()` writes run metadata after each stage
- [x] **All checkpoints .ckpt**: unified via Lightning `ModelCheckpoint` + `load_from_checkpoint()`
- [x] **Fail-fast resources**: `manifest.py` raises `ValueError` on missing profile (was silent 16G fallback → 65 OOM kills)
- [x] **Old corrupted runs cleaned**: removed all Run 001/002 dirs from ESS
- [x] **Pre-existing bugs fixed**: `encoder_targets` NameError in VGAE decoder, `math.lerp` (Python 3.13+ only)

**Verify after Run 003 completes:**
- [ ] Each ablation config produces a unique run directory (hash suffix)
- [ ] Shared upstream stages (VGAE autoencoder) are not duplicated
- [ ] DuckDB catalog has rows with `identity_hash IS NOT NULL`
- [ ] `metrics.json` exists in evaluation run dirs
- [ ] VRAM utilization improved (target: 8-12 GB of 16 GB with batch_size=8192)
- [ ] No timeouts at 240 min wall time
- [ ] GPS conv_gps jobs complete without OOM (VRAM-aware cap ~20K nodes)
- [ ] DGI (unsup_dgi) trains and evaluates successfully
- [ ] `load_from_checkpoint()` round-trips correctly at stage boundaries

**Status tracking:** `sacct -u $USER --starttime=<submit_time>`

### Configs (18 runnable)

| Claim | Configs | What varies |
|-------|---------|-------------|
| Loss × Curriculum | 6 | ce/focal/weighted_ce × curriculum/normal |
| Fusion method | 4 | bandit/dqn/mlp/weighted_avg |
| Conv type | 3 | gatv2/gatv1/gps |
| Unsup method | 3 | vgae/gae/dgi |
| Single-model baselines | 2 | vgae_only/gat_only |

### Deferred

- **KD & scale** (`kd_student`, `large_reference`) — needs `small_kd` preset + teacher wiring

## Recently Completed

- **Checkpoint standardization**: all 7 Lightning modules use `save_hyperparameters()` + `load_from_checkpoint()`. Deleted 4 custom `save_checkpoint()`, 3 `from_checkpoint()`, `load_frozen_cfg()`, envelope sniffing, `_save_and_cleanup()` branching.
- **VRAM-aware batching**: `compute_node_budget()` accepts conv_type/heads, auto-caps GPS via `sqrt(VRAM * 0.6 / cost_per_n2)`.
- Ablation model variants: GAE flag, DGI model + module + eval
- `gat_stage` checkpoint routing for normal vs curriculum
- Data staging: dataset-scoped + skip-tmpdir flags

## In Progress

- Ablation Run 003 (submitting 2026-03-25)
- Ops dashboard (`buckeyeguy/kd-gat-dashboard`) — running on HF Spaces

## Blocked

(none)

## 3-Pillar Architecture (target)

| Pillar | Owner | Current state |
|--------|-------|---------------|
| **Config** | Hydra Compose + Pydantic | **Done** — 5-file config layer + ManifestBuilder |
| **Orchestration** | submitit + graphlib | **Done** — manifest-driven SLURM DAG with stage deduplication |
| **ML Training** | Lightning modules + stages | **Done** — VGAE/GAE/DGI/GAT/fusion models |
| **I/O** | Lightning ModelCheckpoint + DuckDB catalog | **Done** — identity-aware paths, uniform .ckpt, DuckDB append |

## Open Questions

- ~~**GAE vs DGI**~~ **Resolved**: GAE = flag on VGAE, DGI = separate model_type
- ~~**Eval without fusion**~~ **Resolved**: checkpoint-presence guard + `gat_stage` routing
- ~~**Run dir collisions**~~ **Resolved**: `identity_hash` OmegaConf resolver
- ~~**GPS OOM**~~ **Resolved**: VRAM-aware budget auto-caps quadratic convs
- ~~**Checkpoint fragility**~~ **Resolved**: unified Lightning `load_from_checkpoint()` API

## Key Reference Documents

- `ablation.yaml` — 18-config experiment manifest (built by `scripts/build_ablation.py`)
- `plans/ablation-run-001.md` — Run 001 post-mortem with efficiency analysis
- `plans/ablation-001-training-efficiency.md` — Research: VRAM, GPS OOM, data staging
- `graphids/pipeline/manifest.py` — orchestrator
- `graphids/config/pipeline.yaml` — DAG topology + identity_keys
- `graphids/config/resources.yaml` — SLURM resource profiles
