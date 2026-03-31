# KD-GAT Session Plan

> Last updated: 2026-03-30 (session 3)

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

### IO Inconsistencies — RESOLVED

All write paths consolidated into `graphids/config/write_paths.yaml` (single source of truth).
See `plans/architecture/write-paths.md` for full inventory. Remaining cosmetic:

- `production` path prefix in docs never used (always `dev/{user}`)
- `.env.example` doesn't include `KD_GAT_LAKE_ROOT`

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

- [ ] Preprocessing consolidation (`plans/architecture/preprocessing-consolidation.md`) -- delete \_temporal.py, DataModule convention fixes

## Recently Completed

### Write-path consolidation (2026-03-30)

All runtime write paths declared in `graphids/config/write_paths.yaml`, loaded by `config/__init__.py`,
consumed as constants (`CKPT_SUBPATH`, `LAST_CKPT_SUBPATH`, `COMPLETE_MARKER`, etc.). No hardcoded
path strings in Python code.

Key changes:
- `write_paths.yaml` — single source of truth for all disk writes
- `__main__.py` — thin dispatcher for all subcommands (fit/analyze/profile/run/validate-recipe)
- `cli.py` — slimmed: `GraphIDSCLI` + `WandbSaveConfigCallback` + checkpoint dirpath pin + CSVLogger save_dir patch
- `orchestrate/__main__.py` — deleted (absorbed into `__main__.py`)
- Checkpoints decoupled from CSVLogger versioning (`{run_dir}/checkpoints/`, not `lightning_logs/version_N/`)
- Optimizer wiring removed from CLI (models own `configure_optimizers`)
- CurriculumEpochCallback moved to `curriculum.yaml` (not in shared `trainer.yaml`)
- MLflow references cleaned (`.gitignore`, `ci.yml`, `data_loader.py`, stale skills)
- `dagster-ui.sh` sources `.env` for `DAGSTER_HOME`

Full inventory: `plans/architecture/write-paths.md`

### Dagster test suite + test quality pass (2026-03-30)

**Production fix:** `component.py` — DRY_RUN returns early before `complete_marker.touch()` (was crashing
on non-existent directory). Extracted `build_cli_args()` as pure function from `_train` closure.

### Models consolidation complete (2026-03-30)

### DQN/Bandit Lightning conversion (2026-03-30)

Converted DQN and Bandit fusion agents from plain Python classes to proper LightningModules
(models-consolidation.md §8-10). `FusionModuleBase` provides shared base. `RLFusionModule`
deleted. `reward_kwargs_from_cfg()` deleted (dead code). Backward-compat aliases retained.
New stage YAML: `fusion_dqn.yaml`. Registry updated. See `plans/architecture/models-consolidation.md`.

### Observability fully wired (2026-03-30)

All 5 profiling/observability tools wired and verified:

| Tool                    | What it provides                     | Config                                          |
| ----------------------- | ------------------------------------ | ----------------------------------------------- |
| wandb (pynvml)          | GPU util%, temp, power, memory (15s) | `trainer.yaml` WandbLogger                      |
| DeviceStatsMonitor      | CUDA allocator stats per step        | `trainer.yaml` callback                         |
| PyTorchProfiler         | Op-level timing, chrome traces       | `overlays/profile.yaml` + `profile_training.sh` |
| WandbSaveConfigCallback | Full jsonargparse config to wandb    | `cli.py` `save_config_callback`                 |
| sacct profiler          | RSS, CPU%, wall time, mem efficiency | `python -m graphids profile`                    |

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
4. **DynamicBatchSampler num_steps** — CurriculumSampler.\_build_inner() didn't pass
   num_steps (val loader did). len() undefined error.
5. **CurriculumEpochCallback** — set_epoch() was only called at epoch 0 because
   reload_dataloaders_every_n_epochs defaults to 0. Added callback in trainer.yaml.
6. **Fusion YAML routing** — per-method stage YAMLs (fusion_mlp.yaml,
   fusion_weighted_avg.yaml) + identity metadata params on MLPFusionModule/WeightedAvgModule.
7. **MLPFusionModule.state_dim** — required param with no default. Now defaults to
   fusion_state_dim().
8. **Smoke seed collision** — default seed 0 (was 42, same as production).

### Dagster orchestration rebuild (2026-03-29)

Replaced custom orchestration with dagster primitives: `SlurmTrainingComponent`, `CheckpointPathIOManager`,
`SlurmTrainingResource`, asset factory (`enumerate_assets` + `_make_asset`). See `plans/architecture/dagster-native-orchestration.md`.

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

Single dispatcher for all subcommands:

```bash
python -m graphids fit --config graphids/config/stages/autoencoder.yaml
python -m graphids analyze --config graphids/config/stages/analyze_vgae.yaml \
    --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa
python -m graphids profile <job_ids>
python -m graphids run [recipe args]
python -m graphids validate-recipe [recipe args]
```

### Config layout (`graphids/config/`)

```
__init__.py          # constants, topology, path helpers, write-path constants
write_paths.yaml     # single source of truth for all runtime write paths
constants.yaml       # static values
pipeline.yaml        # DAG topology: stages, dependencies, identity_keys
datasets.yaml        # dataset catalog (YAML anchors)
resources.yaml       # SLURM resource profiles
trainer.yaml         # default_config_files: seed, trainer, loggers, callbacks
stages/              # one per stage + analyze configs
overlays/            # thin scale/ablation variants
```

### Orchestration (`graphids/orchestrate/`)

```
orchestrate/
  __init__.py          # package docstring
  component.py         # SlurmTrainingComponent + CheckpointPathIOManager +
                       #   SlurmTrainingResource + enumerate_assets() + _make_asset()
  definitions.py       # dagster entry point (build_defs_for_component)
  run.py               # dagster launch via dg CLI
  validate.py          # recipe config chain validation
  slurm.py             # sbatch submit, sacct poll, script gen
  resources.py         # ResourceSpec + scale_resources (reads resources.yaml)
  profiler.py          # sacct resource profiler (RSS, CPU%, wall time)
```

### Key Reference Documents

- `plans/architecture/write-paths.md` -- **active**: full write path inventory, execution order, conflict table
- `plans/architecture/dagster-native-orchestration.md` -- **active**: dagster primitives, Component, IOManager
- `plans/architecture/dagster-history.md` -- archived: timeline, lessons, postmortem from dagster build
- `plans/experiment-sweep-plan.md` -- ablation claims, configs, stage sharing DAG
- `plans/tier-priority-and-implementation.md` -- priority-ordered task list
- `plans/architecture/models-consolidation.md` -- **completed**: GraphModuleBase, optimizer wiring, cleanup
- `plans/architecture/preprocessing-consolidation.md` -- deferred: delete \_temporal.py, DataModule fixes
- `plans/architecture/flatten-model-config.md` -- completed: config flatten reference
- `plans/architecture/trainer-yaml-wiring.md` -- completed: trainer.yaml verification
- `plans/research/profiling-and-observability.md` -- **active**: consolidated profiling, optimization, observability plan
- `graphids/config/pipeline.yaml` -- DAG topology + identity_keys
- `graphids/config/resources.yaml` -- SLURM resource profiles
