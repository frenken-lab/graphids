# Dagster Orchestration — History & Lessons

> Consolidated from dagster-integration.md + orchestrate-rewrite.md + dagster-ablation-postmortem.md
> Current architecture: `dagster-native-orchestration.md`
> Run 004 failures: `../ablation-run-004-failures.md`
> Audited: 2026-03-30

## Timeline

| Date | Commit | Event | Outcome |
|------|--------|-------|---------|
| 03-27 | `191fe8a` | Wire LightningCLI config, add orchestrator + resource profiles | Initial `dagster_defs.py` |
| 03-28 | `faaac1e` | Flatten all LightningModule configs to typed primitives | Config flatten done |
| 03-28 | `bfb1249` | Wire KD configs, fix 5 stale-ref bugs | Config validation hardened |
| 03-28 | — | Evaluate dagster-slurm: rejected (SSH claim, later corrected) | See dagster-slurm section |
| 03-28 | — | Phase B spike: slurm.py + dagster_defs.py | gpudebug job 46121143 COMPLETED |
| 03-28 | — | trainer.yaml wired as default_config_files | All stages get callbacks, mixed precision |
| 03-29 | `c8bf886` | Collapse expand.py into dagster_defs, add dagster-slurm dep | expand.py deleted |
| 03-29 | `13d419f` | **Rebuild dagster orchestration with proper primitives** | dagster_defs.py → component.py + definitions.py. CheckpointPathIOManager, SlurmTrainingResource, SlurmTrainingComponent. 32 assets, 32 checks. |
| 03-29 | `21060bb` | Fix 8 runtime bugs from smoke test | lake_root defaults, SaveConfigCallback, CurriculumDataModule inheritance, batch sampler, epoch callback, fusion routing, MLP state_dim, smoke seed |
| 03-29 | `c56da05` | Fix asset check: partitions_def, closure capture | |
| 03-29 | `94884cc` | Separate identity keys from model overrides via model_keys | |
| 03-29 | `eb9b416` | Move recipes to config/recipes/, add --recipe flag | RECIPE_PATH = `config/recipes/ablation.yaml` |
| 03-29 | `418a641` | Fix run CLI: require --dataset/--partition, pin alembic | |
| 03-30 | — | **Run 004 submitted** (ablation.yaml, set_01/set_02, seed 42) | **100% failure** — 0/36 stages completed across 2 attempts |
| 03-30 | `eac2a43` | Fix run 004 failures: RAM profiles, dagster logging, .complete markers | 3 of 5 issues fixed |
| 03-30 | `f1ff51b` | Add observability: wandb + DeviceStatsMonitor, VRAM probe | Addresses issue #6 (zero observability) |
| 03-30 | — | wandb fully wired: WandbSaveConfigCallback, _preamble.sh env vars | Issue #6 fully resolved |
| 03-30 | — | Dagster UI: webserver + daemon launcher (`scripts/dev/dagster-ui.sh`) | Orchestration observability ready |

## Run 004 — failure summary

Two orchestrator attempts, both failed. See `../ablation-run-004-failures.md` for full details.

| # | Issue | Jobs hit | Status |
|---|-------|----------|--------|
| 1 | SLURM RAM OOM (24G insufficient for set_01) | 6 | **Fixed** — bumped to 36G |
| 2 | dagster `context.log.warning()` TypeError (structlog kwargs) | 6 | **Fixed** — switched to f-string |
| 3 | Large GAT CUDA OOM (`vram_node_budget()` model-blind) | 1 | **Open** — needs arch-aware budget |
| 4 | KD autoencoder wall time (teacher VRAM collapses batch 3x) | 1 | **Open** — needs KD resource profiles + teacher VRAM fix |
| 5 | `profile_jobs.py` broken for dagster log layout | — | **Open** — profiler expects submitit paths |
| 6 | Zero observability (no logger, no GPU stats, no epoch progress) | all | **Fixed** — wandb + DeviceStatsMonitor + WandbSaveConfigCallback + _preamble.sh env vars + dagster UI |

### Pre-submission bugs (caught during development, before Run 004)

1. `--print_config` null serialization: `Optional[X]=None` → `null` overrides Python defaults. Fix: explicit values in stage YAMLs.
2. LearningRateMonitor + `logger:false`: Lightning raises at `on_train_start`. Fix: removed LRM from trainer.yaml.
3. `pool_aggrs=None` in GATWithJK: no None guard at `len()`. Fix: `pool_aggrs = pool_aggrs or ("mean",)`.

### Teacher VRAM root cause (issue #4)

Lightning auto-moves `self.teacher` (child `nn.Module`) to GPU at setup. `offload_teacher_to_cpu` is doubly broken: (1) not a declared `__init__` param — `getattr(..., False)` never fires; (2) Lightning already moved teacher to GPU. The 11 GiB is activation memory from forward pass on 168K nodes through large VGAE (`[480,240,64]`, 4 heads), not parameter memory (~3 MB). Fix options: register teacher as non-module attribute, or use `configure_model()` hook.

**Run 005 not yet submitted.** Issues #3 and #4 block resubmission — large models will OOM again.

## Execution model

Single CPU SLURM job runs dagster orchestrator. Submits GPU training jobs via sbatch, polls via sacct.
- Entry: `python -m graphids.orchestrate run` → `dg launch` subprocess
- Definitions discovered via `pyproject.toml` `code_location_target_module` → `definitions.py`
- `SlurmTrainingComponent.build_defs()` generates assets from `pipeline.yaml` + `recipes/ablation.yaml`
- `multiprocess_executor(max_concurrent=8)` fans out independent assets
- Restart-safe: skip requires both `best_model.ckpt` AND `.complete` marker (not just any checkpoint)
- `DAGSTER_HOME=/fs/scratch/PAS1266/dagster` (SQLite run history).
Webserver + daemon: `bash scripts/dev/dagster-ui.sh` (port 3000, SSH tunnel for local access)

## Dagster vs Lightning responsibility split

| Layer | Owns | Coupling point |
|-------|------|----------------|
| **Dagster** | DAG ordering, partitions (dataset×seed), retry (OOM→scale_resources), skip-if-done, metrics metadata | Filesystem path convention |
| **Lightning** | Model init, training loop, checkpointing, wall-time requeue (USR1) | `python -m graphids fit --config ...` |
| **Contract** | sbatch script: `_preamble.sh` → training command → `_epilog.sh` | `{lake_root}/dev/{user}/{dataset}/{model}_{scale}_{stage}_{hash}/seed_{N}/` |
| **IOManager** | Checkpoint path handoff between stages via JSON sidecars at `{lake_root}/.dagster/io/` | Upstream asset returns ckpt path string |

## Decisions

| Decision | Rationale | Status |
|----------|-----------|--------|
| ~~No daemon/webserver~~ Webserver + daemon on login node | `scripts/dev/dagster-ui.sh` starts both. SSH tunnel for browser access. | **Updated** 2026-03-30 |
| `MultiPartitionsDefinition(dataset, seed)` | Covers sweep matrix without custom code | Valid — implemented |
| dagster-slurm rejected | Pipes protocol overhead, remote-first design, slurm.py not the problem | **Confirmed** — custom `SlurmTrainingResource` used instead. dagster-slurm unused dep in pyproject.toml (remove it). |
| Dagster Pipes rejected | Training is CLI commands, not Pipes-aware Python | Valid — revisit if in-job metric streaming needed |
| Custom CheckpointPathIOManager | JSON sidecar for ckpt path handoff between stages | **Implemented** — `component.py:57-84` |
| dagster Component (`dg.Component`) | YAML-driven config, `dg` CLI discovery, scaffolding | **Implemented** — `SlurmTrainingComponent` at `component.py:427` |
| expand.py for config serialization | Recipe-direct loading is simpler | **Superseded** — `enumerate_assets()` reads recipe inline |
| .complete marker for skip-if-done | `best_model.ckpt` alone insufficient (stale/killed runs) | **Implemented** — `eac2a43` |

## Test gaps (still open)

None of these tests exist yet (verified 2026-03-30):

| Test | Layer | Purpose |
|------|-------|---------|
| `test_recipe_configs_parse` | 0 (pure Python) | All 18 ablation configs pass jsonargparse |
| `test_gatwjk_pool_aggrs_none` | 0 (pure Python) | GATWithJK handles `pool_aggrs=None` |
| `test_trainer_yaml_callbacks_compatible` | 0 (pure Python) | trainer.yaml callbacks work with `logger:false` |
| `test_dagster_defs_load` | 1 (dagster unit) | `dg list defs` / `dg check defs` equivalent |
| `test_checkpoint_io_manager` | 3 (IOManager) | Sidecar write/read round-trip |
| `test_enumerate_assets_count` | 0 (pure Python) | 32 assets from ablation recipe |

Dagster testing layers (from docs audit):
- **Layer 0**: Pure Python — `compute_identity_hash()`, `run_dir()`, `enumerate_assets()`, CLI arg building
- **Layer 1**: Dagster unit — direct-invoke assets with `mock.Mock(spec=SlurmTrainingResource)`
- **Layer 2**: Dagster integration — `materialize_to_memory()` with fake SLURM + real IOManager
- **Layer 3**: IOManager unit — `build_output_context()` / `build_input_context()`
- **Layer 4**: Smoke on gpudebug — `python -m graphids.orchestrate smoke`

## Observability roadmap

| Fix | Effort | Value | Status |
|-----|--------|-------|--------|
| `python_logs` file handler in `dagster.yaml` | Config only | Catches subprocess TypeError tracebacks | **Open** |
| `dagster debug export <run_id>` | Zero — already available | Failed run inspection | Never used |
| wandb + DeviceStatsMonitor in Lightning | Done | Loss curves, GPU stats | **Implemented** (`f1ff51b`) + WandbSaveConfigCallback + env vars (2026-03-30) |
| `AssetObservation` in poll loop | ~10 lines | SLURM state transitions in dagster events | **Open** |
| `context.add_output_metadata()` | ~5 lines/asset | job_id, wall time, peak RSS on materialization | **Open** |
| Dagster Pipes for in-job metrics | Significant | Epoch progress mid-job | Deferred |

## Config system (reference)

jsonargparse + flat YAML. CLI: `python -m graphids fit --config stages/X.yaml --config overlays/Y.yaml --model.init_args.foo=bar`. See `flatten-model-config.md`.

Ablation recipe: `graphids/config/recipes/ablation.yaml` (83 lines, 18 configs, 6 paper claims). Sweep: 2 datasets × 1 seed. `RECIPE_PATH` env-overridable via `KD_GAT_RECIPE`.
