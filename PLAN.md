# KD-GAT Session Plan

> Last updated: 2026-03-31 (end of config/model/orchestrate refactor session)

## Current State

Run 005 completed 22/36 jobs on Ascend A100. All 3 immediate failures fixed (fusion wiring, preprocessing OOM, wall times). **NOT ready to relaunch** — systemic issues remain.

Config, model, and orchestrate layers fully refactored (see Handoff below). Documentation updated to match new structure. Public API (`graphids.config` imports) unchanged.

## Handoff — What the last session did (2026-03-31)

Config/model/orchestrate refactor, building on prior scripts/ cleanup.

### Config decomposition

Monolithic YAML files split into focused modules:
- `constants.yaml` → `runtime.py` (env vars) + `defaults/global.yaml` + `defaults/io.yaml`
- `pipeline.yaml` → `topology.py` (Python, import-time validation) + `matrix/axes.yaml`
- `datasets.yaml` → `config/datasets/*.yaml` (per-dataset) + `load_catalog()` in `paths.py`
- `resources.yaml` → `config/resources/profiles/*.yaml` + `clusters.yaml` + `submit_profiles.yaml`
- `trainer.yaml` → `config/defaults/trainer.yaml`
- `write_paths.yaml` → removed (logic in `paths.py`)
- `overlays/` → removed; model configs now use `models/{family}/base.yaml` + `scales/{scale}.yaml`

New Python modules: `base.py` (CONFIG_DIR, PROJECT_ROOT), `runtime.py`, `topology.py`, `paths.py`, `contracts.py`, `yaml_utils.py`, `recipe_expand.py`. `__init__.py` is a re-export facade — public API unchanged.

### Fusion config consolidation

Per-method stage YAMLs (`fusion_bandit.yaml`, `fusion_dqn.yaml`, `fusion_mlp.yaml`, `fusion_weighted_avg.yaml`) replaced with single `stages/fusion.yaml` + per-method overlays in `config/fusion/methods/{method}.yaml` + fusion scales in `config/fusion/scales/`.

### Model directory restructure

Flat `core/models/*.py` reorganized into family packages:
- `vgae.py`, `dgi.py` → `autoencoder/`
- `gat.py` → `supervised/`
- `temporal.py` → `temporal_family/`
- `bandit.py`, `dqn.py`, `fusion_baselines.py`, `fusion_features.py`, `fusion_reward.py` → `fusion/`

Shared utilities remain at top level: `_conv.py`, `_training.py`.

### Orchestrate decomposition

Monolithic orchestrate split into: `planning.py`, `execution.py`, `assets.py`, `checks.py`, `analysis.py`, `slurm.py`, `validate.py`. `component.py` is the integration hub. Contracts moved to `core/contracts/` (`analysis.py`, `models.py`, `ops.py`).

### CLI changes

- Commands use explicit `_COMMAND_MODULES` dict in `__main__.py`, NOT auto-discovery. Adding a subcommand = one file + one dict entry.
- New commands: `train-from-spec`, `analyze-from-spec` (spec-file transport for dagster→SLURM).

### Prior session (scripts refactor)

scripts/ reduced from 20 files to 3: `submit.sh`, `slurm/_preamble.sh`, `slurm/_epilog.sh`. All job logic in Python CLI subcommands. Separated dagster from CLI (`dg launch` directly, no wrapper).

### Stale memories identified but NOT cleaned

`project_hydra_config_refactor.md` (Hydra was rejected), `feedback_yaml_only_config.md`, `feedback_never_run_tests_login.md`, `feedback_slurm_partition.md` all duplicate rules files. Delete these.

## Blocking — Must fix before relaunch

1. **Forced callbacks** — ModelCheckpoint silently dropped by jsonargparse list replacement. Curriculum runs trained 300 epochs with no checkpoint. Fix spec ready: `plans/architecture/forced-callbacks.md`
2. **Open issues triage** — `plans/open_issues.md` has 25+ deferred items across 6 categories. Need to prioritize: which block correct results vs which are cleanup.

## In Progress

- HF Spaces dashboard (`buckeyeguy/kd-gat-dashboard`)

## Next (after relaunch succeeds)

1. Evaluation + analysis as dagster assets (`plans/architecture/evaluation-analysis-assets.md`)
2. HPO sweep with Optuna (Phase 2 of `plans/experiment-sweep-plan.md`)
3. Final evaluation — best config, all 6 datasets, 3+ seeds (Phase 3)

## Key References

| Doc | Purpose |
|-----|---------|
| `plans/open_issues.md` | All deferred items, consolidated |
| `plans/experiment-sweep-plan.md` | 17-config ablation matrix, stage-sharing DAG, phased HPO |
| `plans/ablation_and_main_005.md` | Run 005 job summary + failure post-mortem |
| `plans/architecture/forced-callbacks.md` | Checkpoint loss fix — ready to implement |
| `plans/scripts-refactor-option-c.md` | scripts/ cleanup — thin shells, Python does the work |
| `plans/architecture/write-paths.md` | Filesystem layout, write path inventory |
| `plans/architecture/dagster-native-orchestration.md` | Component architecture reference |
| `plans/research/profiling-and-observability.md` | What's wired, what's not, tool decisions |
