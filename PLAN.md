# KD-GAT Session Plan

> Last updated: 2026-03-28

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

### IO Inconsistencies

Spikes, Slurm logs, tests, and profiles still write to this repo and not to share lake folder

### Configs (18 runnable)

| Claim                  | Configs | What varies                              |
| ---------------------- | ------- | ---------------------------------------- |
| Loss × Curriculum      | 6       | ce/focal/weighted_ce × curriculum/normal |
| Fusion method          | 4       | bandit/dqn/mlp/weighted_avg              |
| Conv type              | 3       | gatv2/gatv1/gps                          |
| Unsup method           | 3       | vgae/gae/dgi                             |
| Single-model baselines | 2       | vgae_only/gat_only                       |

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

### Artifacts `analyze` subcommand (2026-03-28) — DONE

`python -m graphids analyze --config stages/analyze_vgae.yaml --analyzer.ckpt_path ... --analyzer.dataset ...`. Same jsonargparse, YAML under `analyzer:` namespace. Deleted dead `generate_all`, rewrote 4 leaf modules to take concrete inputs, created `Analyzer` class. Fail-loud on missing checkpoints/deps. See `plans/fluffy-floating-galaxy.md`.

### Config flatten + consolidation (2026-03-28) — DONE

Replaced Hydra/OmegaConf + config dataclasses with jsonargparse + flat YAML. All 5 LightningModules take flat typed primitives. Deleted `schema.py`, `coerce_config`, `resolve()`, `defaults/` directory. Flattened `pipeline.yaml` identity keys, gutted dead registry arch factory, flattened `CurriculumSampler`, added checkpoint migration guard. See `plans/flatten-model-config.md`.

### Lightning callback extraction (2026-03-27) — DONE

Replaced handrolled runner.py orchestration with Lightning callbacks + LightningCLI.

**New files:**

- `graphids/pipeline/callbacks.py` — RunDirectorySetup, PopulateAndBuild, DuckDBCatalog
- `graphids/pipeline/cli.py` — GraphIDSCLI (LightningCLI subclass, Trainer-from-config)

**Refactored:**

- runner.py: 359→238 lines. `_train` delegates to `cli.train_stage`
- Modules: deferred `build_model()`, self-resolving KD teacher via `prepare_kd`
- CurriculumDataModule: self-contained `from_cfg()` (builds raw DM + VGAE internally)
- Configurable callbacks: `swa_enabled`, `device_stats`, `lr_monitor` in TrainingConfig
- Shared `find_threshold()` extracted from VGAE/DGI into `_training.py`

### Config system rewrite (2026-03-26) — DONE

Replaced Hydra/OmegaConf with jsonargparse + plain YAML. Then reorganized
the entire config package for single-responsibility and data/logic separation.

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

**GPU profile (Run 003, 2026-03-25):**

| Model | Dataset | GPU util (training) | VRAM peak | CPU RSS | Time |
| ----- | ------- | ------------------- | --------- | ------- | ---- |
| VGAE  | set_01  | 83%                 | 8.8G/16G  | 13G     | 1:12 |
| VGAE  | set_02  | 77%                 | 10.0G/16G | 22G\*   | 1:16 |
| GAT   | set_02  | 90%                 | 13.1G/16G | 13G     | 2:11 |

\*VGAE/set_02 high RSS is CPU-side worker memory bloat, not GPU. 19% idle GPU = DataLoader-bound. PrefetchLoader should help on next run.

## Blocked

(none)

## Open Questions

- VGAE worker memory bloat (13G vs 22G bimodal) — same model, different nodes. PrefetchLoader may help, needs rerun to confirm.
- `--mem` over-requesting 54G when peak is 23G — update `resources.yaml` to 32G after confirming PrefetchLoader doesn't change RSS profile.

## Key Reference Documents

- `ablation.yaml` — 18-config experiment manifest (built by `python -m graphids build ablation`)
- `plans/ablation-run-001.md` — Run 001 post-mortem with efficiency analysis
- `plans/ablation-001-training-efficiency.md` — Research: VRAM, GPS OOM, data staging
- `graphids/pipeline/manifest.py` — orchestrator
- `graphids/config/pipeline.yaml` — DAG topology + identity_keys
- `graphids/config/resources.yaml` — SLURM resource profiles
