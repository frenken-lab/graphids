# Hydra Sweep Plugins: Optuna Sweeper + Submitit Launcher

> Created: 2026-03-22

## Context

Hyperparameter sweeps currently use Hydra `--multirun` with the default basic sweeper (grid over comma-separated values) and default local launcher (sequential in-process inside a single SLURM job). This means:
- No smart search (no Bayesian optimization, no early pruning)
- No parallelism (all trials run one after another in one job)
- Wall time = sum of all trials, not max

We installed two Hydra plugins:
- `hydra-optuna-sweeper` 1.1.2 — replaces grid sweep with Optuna TPE/CMA-ES
- `hydra-submitit-launcher` 1.2.0 — replaces local launcher with SLURM job submission

Combined: Optuna picks hyperparameters, submitit launches each trial as a SLURM job (via job arrays internally), multiple trials run on separate GPUs in parallel.

## Current Architecture

```
__main__.py:main()
  └─ @hydra.main(config_name="config")
       └─ run(cfg) → run_stage(cfg, stage) → returns val_loss (float)
```

**Already compatible:**
- `main()` returns `val_loss` (line 96) — Optuna needs a scalar objective ✓
- `config.yaml` has a `hydra:` section (line 43) — can extend it ✓
- Config composition works: `config.yaml` → model preset → CLI overrides ✓

**Needs work:**
- No `hydra/sweeper` or `hydra/launcher` override in `config.yaml`
- No sweep-specific config (search spaces, n_trials, parallelism)
- The preset merge in `main()` (lines 85-88) runs inside `run()` — the launcher needs the function to be pickle-safe for SLURM submission

## Design Principles

> These guard against the pattern of writing custom shortcuts instead of composable building blocks.

1. **Use the framework** — Hydra sweeper + launcher APIs, not custom sweep scripts. Optuna's study API for result selection, not hand-rolled result parsing.
2. **Config is the building block** — Search spaces live in reusable YAML files (`config/experiment/`), not in CLI invocations or inline in Python. Resource profiles come from `resources.yaml`, not hardcoded in launcher config.
3. **Compose, don't inline** — Experiment configs compose via Hydra defaults (`+experiment=gat_tuning`). The launcher reads SLURM params from config, not from ad-hoc `sbatch` flags.
4. **One source of truth** — SLURM resource profiles stay in `resources.yaml`. Optuna study names derive from config interpolation. No duplicated constants.
5. **Framework for 80%, thin wrapper for 20%** — Hydra+Optuna+submitit handle trial lifecycle, SLURM submission, Bayesian search. We only write: experiment configs, and (future) the DAG↔sweep bridge in `submit_dag()`.

## Plan

### Task 1: Add launcher + sweeper config to `config.yaml`

Add Hydra overrides. These are **opt-in** — training runs still use local launcher by default. Sweep runs pass `--multirun hydra/launcher=submitit_slurm hydra/sweeper=optuna` on the CLI.

```yaml
# config.yaml additions at the end of hydra: section

hydra:
  # ... existing job/run config ...
  sweeper:
    # Optuna defaults — overridable per experiment
    direction: minimize
    n_trials: 30
    n_jobs: 1       # sequential by default; submitit handles parallelism
    study_name: ${model_type}_${scale}_${stage}
    storage: null   # in-memory; could point to sqlite for persistence
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: ${seed}
      n_startup_trials: 10
    params: {}      # defined per-experiment via config/experiment/*.yaml

  launcher:
    # Submitit SLURM defaults — used only with --multirun hydra/launcher=submitit_slurm
    partition: gpu
    gpus_per_node: 1
    cpus_per_task: 4
    mem_gb: 48
    timeout_min: 480
    account: ${oc.env:KD_GAT_SLURM_ACCOUNT,PAS1266}
    array_parallelism: 4
    setup:
      - "source /users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh"
```

**Building-block check:** SLURM resource defaults here match `resources.yaml` GPU profile. If `resources.yaml` changes, this should follow. Consider: should the launcher read from `resources.yaml` directly? For MVP, static defaults are fine. For future, a Hydra resolver that reads `resources.yaml` would be the clean path.

**Files:** `graphids/config/config.yaml`

### Task 2: Verify `run()` is pickle-safe for submitit

Submitit serializes the job function with cloudpickle and ships it to the SLURM node. The current `run()` is a closure inside `main()` that captures `cli_overrides`. This **may not pickle** because it captures from the enclosing scope.

**Fix options (in order of preference):**
1. **Test first** — submitit uses cloudpickle which handles closures. It might just work.
2. **Extract to module level** — move `run()` out of `main()`, pass `cli_overrides` as a parameter via `functools.partial` or Hydra's config injection.
3. **Do NOT** write a custom serialization wrapper — use cloudpickle (already a submitit dep).

**Verification:** Run a trivial 2-trial sweep on `gpudebug` partition and confirm both trials execute.

**Files:** `graphids/__main__.py` (may need refactoring if pickle fails)

### Task 3: Experiment-specific sweep configs

Rather than passing long CLI strings (which are not reusable, not version-controlled, and not composable), create experiment config files:

```
graphids/config/experiment/
  imbalance_ablation.yaml    # loss_fn, class weights, curriculum schedule
  vgae_architecture.yaml     # hidden_dims, latent_dim, heads, conv_type
  gat_tuning.yaml            # lr, dropout, layers, heads
```

Example `gat_tuning.yaml`:
```yaml
# @package _global_
defaults:
  - override hydra/sweeper: optuna
  - override hydra/launcher: submitit_slurm

stage: curriculum
model_type: gat
scale: large

hydra:
  sweeper:
    n_trials: 50
    params:
      training.lr: range(1e-4, 1e-2, log=true)
      training.dropout: range(0.1, 0.5, step=0.05)
      gat.layers: choice(2, 3, 4)
      gat.heads: choice(4, 8)
      gat.conv_type: choice(gatv2, transformer)
```

Usage: `python -m graphids --multirun +experiment=gat_tuning dataset=hcrl_sa`

**Building-block check:** Each experiment config is a self-contained, composable unit. It declares its own sweeper+launcher overrides, search space, and fixed params. No experiment knowledge is hardcoded in Python. Adding a new experiment = adding a YAML file.

**Files:** `graphids/config/experiment/*.yaml` (new config group)

### Task 4: Optuna study persistence

Default `storage: null` means the study is lost when the launcher exits. For crash recovery and multi-session sweeps:

```yaml
hydra:
  sweeper:
    storage: sqlite:///data/optuna/studies.db
    study_name: ${model_type}_${scale}_${stage}_${dataset}
```

**Building-block check:** Use Optuna's native SQLite storage, not a custom results DB. This pairs with the existing MLflow SQLite backend pattern. Study results are queryable via `optuna.load_study()` — no custom parsing.

**Files:** `graphids/config/config.yaml`

### Task 5: Update PLAN.md and docs

- Document the sweep workflow in PLAN.md
- Add example commands for common sweep patterns
- Note that `--multirun` is required for both plugins to activate

## Execution Order

```
Task 1 (config.yaml) — add sweeper + launcher config
    └─→ Task 2 (pickle safety) — verify/fix __main__.py
         └─→ Smoke test: 2-trial sweep on gpudebug
              └─→ Task 3 (experiment configs) — per-experiment yamls
              └─→ Task 4 (persistence) — sqlite storage
```

Tasks 3-4 are independent and can come later. Tasks 1-2 + smoke test are the minimum viable sweep.

## OSC / SLURM Integration Details

### How submitit maps to SLURM

Submitit translates Hydra trials into SLURM job arrays automatically:
```
Hydra --multirun (N trials)
  → submitit groups into: sbatch --array=0-(N-1)%{array_parallelism}
    → each array task runs one trial on its own GPU
```

Submitit also handles:
- **Signal handling**: catches SIGUSR1 for graceful checkpoint + requeue
- **Auto-requeue**: `slurm_max_num_timeout=3` requeues on wall-time preemption
- **Log paths**: `%j` maps to `%A_%a` for array tasks

### OSC resource limits to respect

| Constraint | Value | Impact |
|---|---|---|
| Max running jobs per user | ~256 | `array_parallelism` must stay well under (4-8 for GPU sweeps) |
| GPU partition max walltime | 48h (Pitzer) | `timeout_min: 480` (8h) is safe default per trial |
| gpudebug partition | 1h max, priority | Smoke tests: `hydra.launcher.partition=gpudebug timeout_min=55` |
| 2x V100 per node | `gpus_per_node: 1` | 1 trial per GPU, 2 trials per node max |
| Core-hours budget (PAS1266) | Finite | `array_parallelism` = burn rate multiplier (4 = 4x) |
| Scratch 90-day purge | Affects temp outputs | `hydra.run.dir` under ESS lake_root (permanent) |

### Optuna + submitit interaction

Optuna's `n_jobs: 1` means it picks trials sequentially. But when combined with submitit,
Hydra submits all N trials at once as a job array — submitit handles parallelism, not Optuna.
The `array_parallelism` parameter controls how many run concurrently on the cluster.

For Bayesian optimization to be effective (each trial informed by prior results), consider
setting `array_parallelism` to a smaller value (2-4) so Optuna has completed results
to learn from. Full parallelism (all trials at once) degrades to random search.

## Gotchas

1. **`mp.set_start_method("spawn")`** in `__main__.py:18` — runs at import time. Submitit forks a new Python process per trial, which re-imports `__main__` and calls `set_start_method` again. The `force=True` flag handles this, but verify no CUDA-before-fork issues.

2. **Preset merge** — `OmegaConf.merge(cfg, preset, OmegaConf.from_dotlist(cli_overrides))` at line 88 uses captured `cli_overrides`. When submitit launches trials, `cli_overrides` must be available. If the closure doesn't pickle, this needs restructuring (see Task 2).

3. **`hydra.job.chdir: true`** — each trial gets its own working directory. Correct for sweep isolation. Verify MLflow tracking URI resolves correctly from the new cwd.

4. **Data staging** — the `setup` list sources `_preamble.sh`, which handles modules, venv, and data staging. Each trial job runs this. Smart caching (`.staged_marker`) means the first trial syncs, the rest skip.

5. **`array_parallelism: 4`** — limits concurrent GPU jobs. 4 concurrent trials = 2 nodes. Adjust based on allocation budget.

6. **Optuna + Hydra version compatibility** — `hydra-optuna-sweeper` 1.1.2 requires Hydra 1.1+. We have 1.3.2. Compatible, but verify parameter syntax (`range()` vs `interval()`).

---

## Future Work: Pipeline + Sweep Integration

> This section describes Scenario 2 — integrating HPO sweeps into the multi-stage pipeline.
> Not for immediate implementation. Design only.

### The Problem

Currently `submit_dag()` and `--multirun` sweeps are separate systems. You either:
- Run the full pipeline with fixed configs (`submit_dag`)
- Sweep one stage in isolation (`--multirun`)

The useful case is sweeping within the pipeline:
```
preprocess
    → autoencoder (fixed config)
        → SWEEP curriculum (50 trials, pick best)  ← job array
            → fusion (using best curriculum checkpoint)
                → evaluation
```

### SLURM Primitives That Enable This

**Array dependency chaining** — SLURM natively waits for an entire array:
```bash
SWEEP_ID=$(sbatch --parsable --array=0-49%4 sweep_curriculum.sh)
SELECT_ID=$(sbatch --parsable --dependency=afterok:$SWEEP_ID select_best.sh)
sbatch --dependency=afterok:$SELECT_ID fusion.sh
```

`afterok:$ARRAY_JOB_ID` waits for ALL array tasks. This is the key primitive.

**`afterany` for fault tolerance** — even if some trials fail, the selection job runs and picks the best from successful trials.

### What `submit_dag()` Would Need

The refactor adds a "sweep node" concept to the DAG:

```python
# Conceptual — not final API
for name in topo_order:
    if name in sweep_stages:
        # Use Hydra+Optuna+submitit for this stage (job array)
        sweep_job = submit_sweep(cfg, stage, search_space, n_trials)
        # Lightweight CPU job: query Optuna DB, copy best checkpoint
        select_job = submit_selection(sweep_job, study_name)
        futures[name] = select_job  # downstream stages depend on selection
    else:
        futures[name] = executor.submit(run_stage, cfg, stage)
```

### Building Blocks Required (do NOT inline)

| Need | Building block | Source |
|---|---|---|
| Search space per stage | `config/experiment/*.yaml` | Hydra config group (Task 3) |
| Trial submission | `hydra-submitit-launcher` | Framework (installed) |
| Bayesian search | `hydra-optuna-sweeper` | Framework (installed) |
| Result persistence | `optuna.load_study(storage=sqlite)` | Optuna API (Task 4) |
| Best trial selection | `study.best_trial.params` | Optuna API — NOT custom parsing |
| Resource profiles | `resources.yaml` | Existing building block |
| SLURM array deps | `afterok:$ARRAY_JOB_ID` | SLURM native |
| Checkpoint path convention | `hydra.run.dir` interpolation | Existing config |

**Anti-patterns to avoid:**
- Writing a custom trial results parser instead of using `optuna.load_study()`
- Hardcoding SLURM params in `submit_dag()` instead of reading `resources.yaml`
- Inlining search spaces in Python instead of experiment YAML configs
- Writing a custom job array submission instead of using submitit's `map_array()`
- Building a custom checkpoint selector instead of using Optuna's `study.best_trial`

### Optuna Pruning with Lightning (future optimization)

Optuna can kill unpromising trials early via `PyTorchLightningPruningCallback`. A trial
showing no improvement after N epochs gets pruned instead of running all 200 epochs.
Saves GPU-hours significantly. Requires adding the callback in `trainer_factory.py`
when sweep mode is detected (via Hydra config flag, not a custom env var).

### Prerequisite

Tasks 1-4 from the main plan must be done first. The pipeline integration builds on:
- Working Optuna sweeper (Task 1)
- Working submitit launcher (Task 2)
- Reusable experiment configs (Task 3)
- Persistent Optuna studies (Task 4 — required for selection job to read results)
