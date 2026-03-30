# KD-GAT Session Plan

> Last updated: 2026-03-30 (session 2)

## Active Plan

### Ablation Run 004 — Ready to submit

Run 003 (Hydra-era, 2026-03-25) checkpoints are incompatible with post-flatten code.
Re-training as Run 004 with all 18 configs including KD (configs 10-11).

18 configs x 2 datasets (set_01, set_02) x 1 seed (42). KD configs now wired.

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

Expanded configs now write to `{lake_root}/expanded/` (ESS on OSC). Remaining:
- Slurm logs still write to `slurm_logs/` in repo
- Test outputs still write to repo

### Configs (18 runnable)

| Claim                  | Configs | What varies                              |
| ---------------------- | ------- | ---------------------------------------- |
| Loss x Curriculum      | 6       | ce/focal/weighted_ce x curriculum/normal |
| Fusion method          | 4       | bandit/dqn/mlp/weighted_avg              |
| Conv type              | 3       | gatv2/gatv1/gps                          |
| Unsup method           | 3       | vgae/gae/dgi                             |
| Single-model baselines | 2       | vgae_only/gat_only                       |

### KD pipeline (configs 10-11)

Config 11 (large reference) trains first. Config 10 (KD student) depends on it — pass
teacher checkpoint path via `--model.init_args.auxiliaries[0].model_path=<path>` at submit time.

## In Progress

- Ops dashboard (`buckeyeguy/kd-gat-dashboard`) -- running on HF Spaces
- **Ablation Run 005** -- dagster orchestration verified end-to-end.
  Validate passes, smoke passes (3-stage chain on gpudebug, hcrl_sa, 3 epochs).
  All Run 004 issues resolved. Observability fully wired. Ready to submit.

### Run 004 fixes applied

- [x] SLURM RAM profiles bumped to 36G (resolved issue #1)
- [x] dagster `context.log.warning` TypeError (resolved issue #2)
- [x] **Probe-based VRAM node budget** (resolved issue #3 — large GAT CUDA OOM).
  Replaced `_BYTES_PER_NODE = 32768` constant with `_probe_bytes_per_node()`:
  runs 1 forward pass on ~2000 nodes at `train_dataloader()` time (model on GPU),
  measures `torch.cuda.max_memory_allocated()`, derives real bytes/node.
  Works for all model × scale × GPU combos. CurriculumDataModule defers budget
  from `setup()` to `train_dataloader()`. `GraphModuleBase._oom_safe_step()` remains as safety net.
  **KD-aware** (2026-03-30): probe now runs `model._step()` (auto-detected) instead of
  `forward()`, capturing teacher VRAM during probe. See `plans/memory-profiling/vram-probe-kd-aware.md`.
  Caveat: `_GRAD_MULTIPLIER=2` overestimates for KD (teacher backward doesn't exist) — safe direction.
- [x] **KD teacher VRAM** (resolved issue #4 — Lightning auto-moves teacher to GPU).
  Teacher stored via `self.__dict__["teacher"]` to bypass `nn.Module._modules` registration.
  `teacher_on_device()` moves teacher CPU→GPU only during inference, then back to CPU.
- [x] **Observability** (resolved issue #5 — fully wired).
  WandbLogger + CSVLogger + DeviceStatsMonitor in `trainer.yaml`.
  `WandbSaveConfigCallback` in `cli.py` (forwards full jsonargparse config to wandb).
  Env vars in `_preamble.sh` (`WANDB_DIR=/fs/scratch/PAS1266/wandb`, `WANDB_DISABLE_GIT`, `WANDB_SILENT`).
  Auth: `~/.netrc` (entity: `frenken-2-the-ohio-state-university`). `orchestrate validate` 18/18 pass.
- [x] **Dagster UI** (resolved — webserver + daemon launcher).
  `scripts/dev/dagster-ui.sh` starts both `dagster-webserver` (port 3000) and `dagster-daemon`.
  Access via `ssh -L 3000:localhost:3000 pitzer.osc.edu` → `http://localhost:3000`.
  Verified: HTTP 200, defs load, daemon starts (all 6 daemon types). Use tmux for persistence.

### Code consolidation

- [x] Models consolidation (`plans/architecture/models-consolidation.md`) -- **complete** (2026-03-30)
  - [x] DQN/Bandit Lightning conversion (§8-10): `FusionModuleBase`, `DQNFusionModule`, `BanditFusionModule`
  - [x] §1-7,§11-13: `GraphModuleBase`, optimizer wiring, dead code, temporal fix, YAML configs, verification
  - Deferred: VGAE `configure_optimizers` (projection params), DGI stage YAML
- [x] `orchestrate validate` 18/18 pass (2026-03-30) — fixed cli.py optimizer/scheduler args,
  BanditFusionModule/DQNFusionModule `state_dim` default, `SystemExit` catch in validate
- [ ] Preprocessing consolidation (`plans/architecture/preprocessing-consolidation.md`) -- delete _temporal.py, DataModule convention fixes

## Recently Completed

### Dagster test suite + test quality pass (2026-03-30)

**New orchestrate tests (63 tests, 4 files, all login-node safe):**
- Layer 0 (`test_pure.py`, 43 tests): `run_dir`, `compute_identity_hash`, `build_cli_args`,
  `enumerate_assets`, `_identity_value`, `_cli_val`, `generate_script`, `ResourceSpec`,
  `_detect_cluster`, `get_resources`, `scale_resources`
- Layer 1 (`test_dagster_unit.py`, 4 tests): dry-run, skip-when-complete, failure, IOManager handoff
- Layer 2 (`test_dagster_integration.py`, 6 tests): 3-stage pipeline, sidecar creation, asset checks
- Layer 3 (`test_iomanager.py`, 6 tests): sidecar write/read/overwrite/round-trip/partition isolation

**Production fix:** `component.py` — DRY_RUN returns early before `complete_marker.touch()` (was crashing
on non-existent directory). Extracted `build_cli_args()` as pure function from `_train` closure.

**Existing test quality fixes:**
- Fixed broken `TestVGAECheckpointRoundtrip` (referenced removed `vgae_cfg.vgae` — dead since config flatten)
- Fixed vacuous `assert x.abs().max() <= 10.0 or True` in `test_features.py`
- Removed stale `lr`, `weight_decay`, `gat_stage` from `conftest.py::base_cfg`
- Added `@pytest.mark.slurm` to 10 Trainer tests across 4 files
- Parametrized duplicate tests: `test_vgae` (2 conv types), `test_gat` (3 loss fns),
  `test_fusion` (MLP/WeightedAvg), `test_integration` (3 num_classes variants)
- Strengthened weak assertions: curriculum `set_epoch`, integration threshold comparison
- `test_fusion` `_make_fusion_batch` + checkpoint tests now use `fusion_state_dim()` not hardcoded `15`
- Registered `dagster` marker in `pyproject.toml`; `-m dagster` / `-m "not dagster"` for selection

### Models consolidation complete (2026-03-30)

All 13 steps of `plans/architecture/models-consolidation.md` done. Key outcomes:
- `GraphModuleBase` in `_training.py` — shared `setup()`, OOM guard, BinaryROC threshold for VGAE/GAT/DGI
- `OOMSkipMixin` deleted (absorbed into `GraphModuleBase`)
- `configure_optimizers` deleted from GAT/DGI — CLI auto-wires via `add_optimizer_args`
- Stage YAMLs (`normal.yaml`, `curriculum.yaml`) have explicit `optimizer:`/`lr_scheduler:` sections
- `soft_label_kd_loss` inlined in `gat.py`, `focal_loss` → `_focal_loss` in `gat.py`
- `temporal.py` checkpoint loading simplified (10→2 lines via `load_inner_model`)
- Dead `fuse()` methods deleted from fusion_baselines.py
- Import-level verification passes; SLURM test run pending

### DQN/Bandit Lightning conversion (2026-03-30)

Converted DQN and Bandit fusion agents from plain Python classes to proper LightningModules
(models-consolidation.md §8-10). `FusionModuleBase` provides shared base. `RLFusionModule`
deleted. `reward_kwargs_from_cfg()` deleted (dead code). Backward-compat aliases retained.
New stage YAML: `fusion_dqn.yaml`. Registry updated. See `plans/architecture/models-consolidation.md`.

### Observability fully wired (2026-03-30)

All 5 profiling/observability tools wired and verified:

| Tool | What it provides | Config |
|------|-----------------|--------|
| wandb (pynvml) | GPU util%, temp, power, memory (15s) | `trainer.yaml` WandbLogger |
| DeviceStatsMonitor | CUDA allocator stats per step | `trainer.yaml` callback |
| PyTorchProfiler | Op-level timing, chrome traces | `overlays/profile.yaml` + `profile_training.sh` |
| WandbSaveConfigCallback | Full jsonargparse config to wandb | `cli.py` `save_config_callback` |
| sacct profiler | RSS, CPU%, wall time, mem efficiency | `python -m graphids profile` |

Key fixes applied:
- `WandbSaveConfigCallback` in `cli.py` (Lightning #19728 workaround)
- `_preamble.sh`: `WANDB_DIR`, `WANDB_DISABLE_GIT`, `WANDB_SILENT`, `mkdir -p`
- `link_arguments` for CSVLogger `save_dir` → follows `default_root_dir` (metrics.csv in run dir)
- `dagster-slurm` removed from `pyproject.toml` (unused dep, -5 packages)
- `dagster-webserver` aligned to 1.12.21
- Dagster UI: `scripts/dev/dagster-ui.sh` (webserver + daemon, port 3000, SSH tunnel)
- Profiler rewrite: `scripts/profile_jobs.py` (459 lines) → `graphids/orchestrate/profiler.py`
  (251 lines). Fixed sacct `.0`→`.batch` bug (RSS was always 0), deleted dead `gpu_stats.csv`
  code, shared `sacct_query()` with `slurm.py`, job name metadata parsing.
- `configs/profile.yaml` (orphaned pre-flatten) → `graphids/config/overlays/profile.yaml`
- `profile_training.sh` rewritten to use `_preamble.sh` + current config paths

### Smoke test verified + 8 bug fixes (2026-03-29)

Dagster runtime path verified end-to-end: validate (all config chains) + smoke
(autoencoder→curriculum→fusion, hcrl_sa, 3 epochs, gpudebug). Bugs found and fixed:

1. **lake_root defaults** — 7 modules used `"experimentruns"` instead of
   `os.environ.get("KD_GAT_LAKE_ROOT")`. Data not found at runtime.
2. **SaveConfigCallback** — `overwrite: True` in CLI_KWARGS for reruns.
3. **CurriculumDataModule** — subclassed CANBusDataModule (was missing num_ids,
   in_channels, num_classes properties needed by GATModule.setup()).
4. **DynamicBatchSampler num_steps** — CurriculumSampler._build_inner() didn't pass
   num_steps (val loader did). len() undefined error.
5. **CurriculumEpochCallback** — set_epoch() was only called at epoch 0 because
   reload_dataloaders_every_n_epochs defaults to 0. Added callback in trainer.yaml.
6. **Fusion YAML routing** — per-method stage YAMLs (fusion_mlp.yaml,
   fusion_weighted_avg.yaml) + identity metadata params on MLPFusionModule/WeightedAvgModule.
7. **MLPFusionModule.state_dim** — required param with no default. Now defaults to
   fusion_state_dim().
8. **Smoke seed collision** — default seed 0 (was 42, same as production).

### Dagster orchestration rebuild (2026-03-29)

Replaced `dagster_defs.py` (513 lines) with dagster-native orchestration using proper
primitives: assets with tags/kinds/groups, `CheckpointPathIOManager` for checkpoint
path handoff, `SlurmTrainingResource` for SLURM submission, asset checks, and
`SlurmTrainingComponent` for definition assembly.

Key design decisions:
- **Assets** represent checkpoints (persistent artifacts), not jobs. Each has tags
  (stage, model_type, scale), kinds (checkpoint/metrics), group_name, description.
- **IOManager** stores/retrieves checkpoint path strings via JSON sidecars. Downstream
  assets receive upstream paths as function parameters via `ins=` + `AssetIn`.
- **Resource** wraps slurm.py submit/poll as dagster-injectable `ConfigurableResource`.
- **Asset checks** — 32 blocking `checkpoint_exists` checks.
- **StageConfig** (dataclass) separates training parameters from asset identity.
- **Convention-based config resolution** replaces `_stage_args()` if/elif chain.
- dagster-slurm dropped (Pipes protocol mismatch for on-cluster use).
- `graphids/components/` deleted — component lives in `graphids/orchestrate/component.py`.

Files:
- `graphids/orchestrate/component.py` (~420 lines) — Component + IOManager + Resource + factory
- `graphids/orchestrate/definitions.py` (17 lines) — entry point
- `graphids/orchestrate/__main__.py` (~210 lines) — CLI: run/validate/smoke
- `graphids/orchestrate/slurm.py` (105 lines) — retained
- `graphids/orchestrate/resources.py` (78 lines) — retained

Verified: `dg check defs`, `dg list defs` (32 assets, 32 checks, all tagged),
`smoke --dry-run` (3-stage chain). IOManager `load_input` wired via `ins=`/`AssetIn`
(parameter-based deps, not `deps=` ordering-only).

### P2.5: Collapse expand.py into dagster_defs.py (2026-03-29)

Eliminated the two-phase expand→manifest→dagster pipeline. `dagster_defs.py` now
reads `ablation.yaml` directly, computes topology and identity hashes in-process
(no torch import at definition time), and builds multi-config SLURM commands.

Deleted: `expand.py` (420 lines), `expanded_dir()`, 64 expanded YAMLs + manifest.json.
Changed: `generate_script` accepts `config_files: list[str]` (multi-config flags),
`run_dir()` added to `config/__init__.py`, `orchestrate/__main__.py` gains
`validate`/`smoke` subcommands.

Net: 420 lines deleted (expand.py), ~80 lines added to dagster_defs.py.
SLURM command now: `python -m graphids fit --config stages/X.yaml --config overlays/Y.yaml --model.init_args.foo=bar`

### Dagster Phase C+D: config expansion + dynamic asset graph (2026-03-29)

Verified `trainer.yaml` wiring (all 4 stages get callbacks, mixed precision).
Fixed fusion identity keys: added `conv_type`, `variational` to prevent incorrect
dedup across conv types/unsup methods. Added `variational` to curriculum identity.
Added identity key metadata params to fusion modules (`RLFusionModule`, now `BanditFusionModule`/`DQNFusionModule`) and `GATModule`.
Wrote `ablation.yaml` (18 configs) + `expand.py` (150 lines) + rewrote `dagster_defs.py`
(175 lines) with dynamic asset factory from manifest topology.

32 unique assets (6 autoencoders, 8 curricula, 3 normals, 15 fusions) × 2 datasets
= 64 expanded YAMLs. DAG deps wired from `STAGE_DEPENDENCIES` + KD cross-pipeline.
Upstream checkpoint paths resolved at materialization time. Dry-run `RUN_SUCCESS`
for `set_01|42` (all 32 assets). Added missing resource profiles (dqn/small/large,
dgi/small). Upgraded alembic 1.6.5→1.18.4 (SQLAlchemy 2.0 compat).

### Dagster orchestrate rewrite + gpudebug spike (2026-03-28)

Replaced `graphids/orchestrate/` with dagster-based system. Deleted `submit.py` (247
lines hand-rolled Pipeline class). New files: `slurm.py` (102 lines, sbatch/sacct),
`dagster_defs.py` (140 lines, asset factory + partitions + retry). Config expansion
via jsonargparse `--print_config`. dagster-slurm rejected (requires SSH). Pipes
protocol rejected (post-hoc metrics sufficient). Added `small` scale resource profiles.
Removed 6 dead hydra packages. See `plans/orchestrate-rewrite.md`.

Gpudebug spike (job 46121143) validated full loop: dagster → sbatch → poll → COMPLETED.
Bugs found and fixed: `link_arguments` for model→data params (`conv_type`, `heads`),
`compute_node_budget` replaced with VRAM-driven `vram_node_budget` (uses
`torch.cuda.mem_get_info`), alembic upgraded for SQLAlchemy 2.0. Discovered
`trainer.yaml` is dead config (not loaded) — blocks Phase C.

### KD wiring + bug fixes (2026-03-28)

Fixed 3 KD bugs: `teacher_on_device` stale nested ref, `prepare_kd` identity hash using
student cfg for teacher path, mixed `.get()`/`getattr()` on hparams. Created 4 overlay YAMLs
(`large_vgae`, `large_gat`, `kd_vgae`, `kd_gat`). Fixed `CATALOG_PATH` missing from config
exports. Fixed stale `"pipeline"` lazy import. Fixed `orchestrate/resources.py` stale path.
Run 003 checkpoints declared incompatible (Hydra-era nested format) — re-training as Run 004.

### Artifacts `analyze` subcommand (2026-03-28)

`python -m graphids analyze --config stages/analyze_vgae.yaml --analyzer.ckpt_path ... --analyzer.dataset ...`. Same jsonargparse, YAML under `analyzer:` namespace. `Analyzer` class in `graphids/core/artifacts/analyzer.py`. Fail-loud on missing checkpoints/deps.

### Config flatten + consolidation (2026-03-28)

Replaced Hydra/OmegaConf + config dataclasses with jsonargparse + flat YAML. All 5 LightningModules take flat typed primitives. Deleted `schema.py`, `coerce_config`, `resolve()`, `defaults/` directory. See `plans/architecture/flatten-model-config.md`.

### Lightning callback extraction + LightningCLI (2026-03-27)

Replaced handrolled runner.py orchestration with Lightning callbacks + LightningCLI. `GraphIDSCLI` in `graphids/__main__.py`. Deleted entire `graphids/pipeline/` package (callbacks.py, cli.py, manifest.py, runner.py, stages/).

### Config system rewrite (2026-03-26)

Replaced Hydra/OmegaConf with jsonargparse + plain YAML. Config package: `__init__.py` (constants + topology + path helpers), `constants.yaml`, `pipeline.yaml`, `datasets.yaml`, `resources.yaml`, `trainer.yaml`, `stages/*.yaml`, `overlays/*.yaml`.

### Codebase cleanup (2026-03-25)

Replaced custom DataLoader/collation/assembly with PyG APIs, adopted Lightning built-ins.

## Blocked

- **Ablation Run 004 eval** -- blocked on training completion. After all jobs finish:
  `python -m graphids test` per run dir, then aggregate results to DuckDB catalog.
- **HPO sweep** -- blocked on ablation results + Optuna integration (Phase 2)
- **Full pipeline** -- blocked on HPO results (Phase 3)

## Open Questions

- VGAE worker memory bloat (13G vs 22G bimodal) -- same model, different nodes. PrefetchLoader may help, needs rerun to confirm.
- `--mem` over-requesting 54G when peak is 23G -- `resources.yaml` updated to 24-32G range. Validate in gpudebug spike.
- ~~dagster-slurm plugin vs custom PipesSlurmClient~~ -- **RESOLVED.** Custom `slurm.py` (99 lines). dagster-slurm requires SSH.

## Current Architecture

### CLI entry points (`graphids/__main__.py`)

```bash
# Training (GraphIDSCLI -> LightningCLI)
python -m graphids fit --config graphids/config/stages/autoencoder.yaml

# Analysis artifacts (Analyzer -- no Trainer)
python -m graphids analyze --config graphids/config/stages/analyze_vgae.yaml \
    --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa
```

### Config layout (`graphids/config/`)

```
__init__.py          # constants, topology, path helpers (single Python file)
constants.yaml       # static values
pipeline.yaml        # DAG topology: stages, dependencies, identity_keys
datasets.yaml        # dataset catalog (YAML anchors)
resources.yaml       # SLURM resource profiles
trainer.yaml         # default_config_files: seed, trainer
stages/              # one per stage + analyze configs
overlays/            # thin scale/ablation variants
```

### Orchestration (`graphids/orchestrate/`)

```
orchestrate/
  __init__.py          # package docstring
  __main__.py          # CLI: run/validate/smoke
  component.py         # SlurmTrainingComponent + CheckpointPathIOManager +
                       #   SlurmTrainingResource + enumerate_assets() + _make_asset()
  definitions.py       # dagster entry point (build_defs_for_component)
  slurm.py             # sbatch submit, sacct poll, script gen
  resources.py         # ResourceSpec + scale_resources (reads resources.yaml)
```

### Key Reference Documents

- `plans/architecture/dagster-native-orchestration.md` -- **active**: replace custom code with dagster-slurm + Component + IOManager
- `plans/architecture/dagster-history.md` -- archived: timeline, lessons, postmortem from dagster build
- `plans/experiment-sweep-plan.md` -- ablation claims, configs, stage sharing DAG
- `plans/tier-priority-and-implementation.md` -- priority-ordered task list
- `plans/architecture/models-consolidation.md` -- **completed**: GraphModuleBase, optimizer wiring, cleanup
- `plans/architecture/preprocessing-consolidation.md` -- deferred: delete _temporal.py, DataModule fixes
- `plans/architecture/flatten-model-config.md` -- completed: config flatten reference
- `plans/architecture/trainer-yaml-wiring.md` -- completed: trainer.yaml verification
- `plans/research/profiling-and-observability.md` -- **active**: consolidated profiling, optimization, observability plan
- `graphids/config/pipeline.yaml` -- DAG topology + identity_keys
- `graphids/config/resources.yaml` -- SLURM resource profiles
