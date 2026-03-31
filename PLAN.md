# KD-GAT Session Plan

> Last updated: 2026-03-31 (end of scripts refactor session)

## Current State

Run 005 completed 22/36 jobs on Ascend A100. All 3 immediate failures fixed (fusion wiring, preprocessing OOM, wall times). **NOT ready to relaunch** — systemic issues remain.

## Handoff — What the last session did (2026-03-31)

Full scripts/ refactor + CLI architecture overhaul. Executed `plans/scripts-refactor-option-c.md` and went further.

**scripts/ reduced from 20 files to 3**: `submit.sh` (unified launcher), `slurm/_preamble.sh`, `slurm/_epilog.sh`. All job logic moved to Python CLI subcommands. Resource profiles read from `config/resources.yaml` (single source of truth).

**CLI restructured into 3 clean entry points**:
- `python -m graphids <cmd>` → training (LightningCLI) + operational (`graphids/commands/`, 8 modules)
- `python -m graphids.orchestrate validate` → dagster config validation
- `dg launch/list/check` → dagster native CLI

**Convention-based dispatch**: `__main__.py` auto-discovers `graphids/commands/<name>.py` modules. Adding a subcommand = one file + one YAML entry. No wiring.

**New Python subcommands**: `rebuild-caches`, `test-preprocessing`, `landscape`, `profile-training`, `stage-data`, `submit-profile`, `analyze` (extracted from `__main__.py`).

**Deleted**: `scripts/lib/` (5 files, zero consumers), 12 SLURM scripts (absorbed into submit.sh/commands/), `scripts/data/stage_data.sh` (→ `commands/stage_data.py`), `scripts/dev/` (dagster-ui, tmux, jupyter).

**Separated dagster from CLI**: `cli/run.py` (dagster wrapper) deleted — use `dg launch` directly. `validate_recipe.py` moved back to `orchestrate/`. `graphids/cli.py` is a single file (GraphIDSCLI class), not a package.

**Stale memories identified but NOT cleaned** — `project_hydra_config_refactor.md` (Hydra was rejected), `feedback_yaml_only_config.md`, `feedback_never_run_tests_login.md`, `feedback_slurm_partition.md` all duplicate rules files. Delete these.

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
