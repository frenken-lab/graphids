# KD-GAT Session Plan

> Last updated: 2026-03-25

## Active Plan

### Ablation Run 003 — Submitting now

18 configs × 2 datasets (set_01, set_02) × 1 seed (42), deduped to 62 SLURM jobs. KD configs deferred.

**Fixes applied (Run 001 + Run 002 post-mortems + 2026-03-25 hardening):**

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

- Ablation Run 003 — training COMPLETED (2026-03-25), eval needs resubmit (weights_only fix)
- Ops dashboard (`buckeyeguy/kd-gat-dashboard`) — running on HF Spaces

### Config system rewrite (2026-03-26) — DONE

Replaced Hydra/OmegaConf with jsonargparse + plain YAML. Then reorganized
the entire config package for single-responsibility and data/logic separation.

**Migration (21 commits):**
- Removed hydra-core, omegaconf, hydra-optuna-sweeper dependencies
- Added jsonargparse[signatures] — handles CLI parsing, type coercion, env vars, YAML merge
- All checkpoint loads now use `weights_only=True` (plain dict hparams)
- Zero omegaconf references remain in codebase

**Dead code removal:**
- 5 wrapper functions deleted (build_optimizer_dict, _get_kd_config, make_test_trainer, _build_callbacks, _nested_get)
- 2 dead parameters removed (load_model stage, make_trainer stage)
- logging.py deleted — structlog.configure inlined at call sites
- Custom _Namespace/_parse_dotlist/_deep_merge replaced by jsonargparse built-ins

**Structural reorganization:**
- `__init__.py` files are re-exports only — all logic moved to named modules
- `config/resolve.py` — config composition (87 lines, single responsibility)
- `config/constants.py` — topology loader, path helpers, identity hash, env vars
- `config/defaults/` — pure values: schema.py, pipeline.yaml, datasets.yaml, resources.yaml, presets.yaml
- Nothing outside `config/` imports from `config.defaults.*` or `config.resolve`

**Single source of truth:**
- `pipeline.yaml` defines valid models, scales, stages → Config defaults derived at import
- `datasets.yaml` uses YAML anchors (set_01–04 were identical copy-paste, now 4 lines each)
- `presets.yaml` uses anchors (vgae/dgi shared unsupervised configs)
- Dataclass schema defines field shapes + types, YAML defines values

**Bug fixes:**
- SLURMEnvironment(auto_requeue=True) wired in make_trainer (auto-save was dead code)
- seed_everything(workers=True) for reproducible DataLoader worker seeding
- Trainer reused across eval scenarios (was creating N+1 instances)

### Codebase cleanup (2026-03-25) — DONE

Replaced custom DataLoader/collation/assembly with PyG APIs, adopted Lightning built-ins.

**Deleted (~700 lines):**
- `_FastCollate`, `_SlicesBatchSampler`, `_IndexDataset` → PyG `DynamicBatchSampler` + standard `DataLoader`
- `_assemble_chunk_numpy`, `_numpy_to_data`, `_assemble_graphs` (ProcessPoolExecutor parallel assembly) → Polars vectorized triangle counting for clustering coefficient + degree
- `_graph_utils.py` (dead), `edge_features` (dead), accumulator pattern in VGAE/DGI
- Dead scheduler config fields, `cache_predictions` as standalone function

**Added:**
- `PrefetchLoader` on train/val DataLoaders (async GPU transfer)
- `LearningRateMonitor` + `StochasticWeightAveraging` callbacks
- `predict_step` + `Trainer.predict()` for threshold search (replaces manual accumulators)
- `afterany` + pre-flight checkpoint check in DAG submission (replaces `afterok` silent cascades)
- Skip-if-done in manifest submission (don't resubmit completed stages)
- DuckDB catalog self-heals missing columns via `ALTER TABLE ADD COLUMN`
- Batch-aware fusion extractors (scatter ops, no `to_data_list()` loops)
- `parse_payload()` moved from can_bus.py to features.py (single source of truth)
- `_load_checkpoint` shared helper for model loading (was duplicated)

**GPU profile (Run 003, 2026-03-25):**

| Model | Dataset | GPU util (training) | VRAM peak | CPU RSS | Time |
|-------|---------|-------------------|-----------|---------|------|
| VGAE | set_01 | 83% | 8.8G/16G | 13G | 1:12 |
| VGAE | set_02 | 77% | 10.0G/16G | 22G* | 1:16 |
| GAT | set_02 | 90% | 13.1G/16G | 13G | 2:11 |

*VGAE/set_02 high RSS is CPU-side worker memory bloat, not GPU. 19% idle GPU = DataLoader-bound. PrefetchLoader should help on next run.

## Blocked

(none)

## Open Questions

- VGAE worker memory bloat (13G vs 22G bimodal) — same model, different nodes. PrefetchLoader may help, needs rerun to confirm.
- `--mem` over-requesting 54G when peak is 23G — update `resources.yaml` to 32G after confirming PrefetchLoader doesn't change RSS profile.

## Key Reference Documents

- `ablation.yaml` — 18-config experiment manifest (built by `scripts/build_ablation.py`)
- `plans/ablation-run-001.md` — Run 001 post-mortem with efficiency analysis
- `plans/ablation-001-training-efficiency.md` — Research: VRAM, GPS OOM, data staging
- `graphids/pipeline/manifest.py` — orchestrator
- `graphids/config/pipeline.yaml` — DAG topology + identity_keys
- `graphids/config/resources.yaml` — SLURM resource profiles
