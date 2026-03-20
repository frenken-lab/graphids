# Replace optuna_sweep.py with hydra-optuna-sweeper

**Date:** 2026-03-20
**Status:** Brainstorm (design-first)
**Depends on:** `codebase-reduction.md` (sections F5, H4a), `stage-executor-and-launcher.research.md` (execute_stage + submitit context)

---

## Problem

> **Ray status:** `ray[tune]` is listed as an optional dep in `pyproject.toml` but nothing in `optuna_sweep.py` imports it. Ray Tune was removed in Phase 2 when Optuna replaced it. The optional extra is a leftover.

`optuna_sweep.py` (302 lines) + `subprocess_utils.py` (72 lines) = 374 lines of custom HPO infrastructure that reimplements what `hydra-optuna-sweeper` provides out of the box. Three layers of translation:

1. **Search space loading** (43-58): Custom YAML → `(type, low, high)` tuples
2. **Parameter suggestion** (71-85): Tuples → `trial.suggest_*()` → Hydra override tuples
3. **Subprocess dispatch** (93-119): Overrides → `build_cli_cmd()` → `subprocess.run()` → read manifest

Each trial spawns a full `python -m graphids.cli` subprocess for CUDA isolation. This is correct behavior — but `hydra-optuna-sweeper` does the same thing natively via Hydra multirun.

### What stays custom today (should stay custom)

- **Warm-start** (`_enqueue_warm_start`, 10 lines): Loads prior sweep YAML and enqueues as initial trial
- **SQLite resume** (`_sweep_db_path`): Built into Optuna regardless of how trials are launched
- **Pipeline sweep** (`run_sweep_pipeline`, 50 lines): Sequential 3-stage sweep → train-best → eval. This is orchestration logic, not HPO
- **Multi-seed final** (`_run_multi_seed_final`, 16 lines): Re-train best config across seeds. Also orchestration

### What should disappear

| Current code | Lines | hydra-optuna-sweeper equivalent |
|-------------|------:|--------------------------------|
| `_load_search_spaces()` | 16 | Sweeper reads from Hydra config natively |
| `_suggest_params()` | 15 | Sweeper calls `trial.suggest_*` from config |
| `_objective()` | 27 | Each trial IS a Hydra run |
| `build_cli_cmd()` | 72 | Not needed — Hydra builds the command |
| `run_sweep()` (Optuna study mgmt) | 52 | `--multirun` + sweeper plugin |
| `_export_best_config()` | 12 | Sweeper logs best params; custom callback for YAML export |
| `load_best_config()` | 6 | Dead code (only called within this file) |
| **Total** | **200** | **~0 Python + YAML config** |

### What transforms (orchestration → submitit or separate concern)

| Current code | Lines | New home |
|-------------|------:|----------|
| `run_sweep_pipeline()` | 33 | Orchestration: `run_dag()` with sweep stage type |
| `_run_multi_seed_final()` | 16 | Orchestration: submitit array job with seed list |
| Warm-start logic | 10 | Hydra sweeper custom search space or callback |

---

## Goals

1. **Zero custom HPO code**: search space definition, parameter suggestion, and trial dispatch handled by hydra-optuna-sweeper
2. **Keep subprocess isolation**: each trial = separate process (CUDA context isolation)
3. **Keep SQLite resume**: Optuna storage for free restart across SLURM preemptions
4. **Keep warm-start**: ability to seed a study from prior sweep results
5. **Preserve CLI UX**: `python -m graphids.cli sweep ...` still works (or better: `python -m graphids.cli --multirun ...`)

## Non-goals

- Multi-objective optimization (single val_loss minimization is sufficient)
- Distributed sweeps across SLURM nodes (one sweep job, sequential trials with subprocess isolation)
- Replacing the pipeline sweep orchestration (that's `run_dag()` territory)

---

## How Hydra and Optuna relate

`hydra-optuna-sweeper` is a **first-party Hydra plugin** (same GitHub org: facebookresearch/hydra). It replaces Hydra's default grid/random sweeper with Optuna's TPE sampler for `--multirun` runs. The relationship:

- **Hydra** owns config composition, CLI parsing, and the multirun loop
- **Optuna** owns the search algorithm (TPE, CMA-ES, etc.), trial storage (SQLite), and pruning
- **hydra-optuna-sweeper** bridges them: reads search spaces from Hydra YAML config, translates to Optuna distributions, runs trials via Hydra's multirun infrastructure

Config is pure YAML (from [hydra.cc/docs/plugins/optuna_sweeper](https://hydra.cc/docs/plugins/optuna_sweeper)):
```yaml
defaults:
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    sampler:
      seed: 123
    direction: minimize
    study_name: my-study
    storage: null  # or sqlite:///path for resume
    n_trials: 20
    n_jobs: 1
    params:
      x: range(-5.5, 5.5, 0.5)
      y: choice(-5, 0, 5)
    # EXPERIMENTAL: custom_search_space for dynamic params
    custom_search_space: .my_module.configure
```

The sweeper calls the `@hydra.main` task function once per trial with the suggested overrides merged into the config. Return value = objective value.

## Key question: Hydra Compose API compatibility

We use `compose_config()` via Hydra Compose API, NOT `@hydra.main`. The sweeper plugin is designed for `@hydra.main` + `--multirun`.

**Confirmed incompatibility:** Hydra docs explicitly state ([hydra.cc/docs/advanced/compose_api](https://hydra.cc/docs/advanced/compose_api)):
> "Avoid using the Compose API in cases where @hydra.main() can be used, as doing so forfeits many of the benefits of Hydra such as Tab completion, **Multirun**, Working directory management, and Logging management."

This means **the sweeper cannot be used from `compose_config()`** — it requires `@hydra.main`. This is not a bug; it's by design. The Compose API is for embedding Hydra in notebooks/tests, not for running sweeps.

### Our setup

```
# cli.py uses compose_config() → PipelineConfig → execute_stage()
# No @hydra.main decorator
# Entry: python -m graphids.cli stage=autoencoder model=vgae_large ...
```

### Implication

We need a **separate entry point** (`sweep.py`) with `@hydra.main` for HPO. `cli.py` continues to use Compose API for single runs. This is two entry points, but they share the same config groups and the same `execute_stage()` function.

**Three possible approaches:**

### Approach A: Thin @hydra.main wrapper (RECOMMENDED)

Add a minimal `sweep.py` entry point that uses `@hydra.main`:

```python
@hydra.main(config_path="config/conf", config_name="config", version_base="1.3")
def sweep_app(cfg: DictConfig) -> float:
    pcfg = PipelineConfig.model_validate(OmegaConf.to_object(cfg))
    stage = cfg.stage
    result = execute_stage(pcfg, stage)
    return result.metrics.get("val_loss", float("inf"))
```

CLI: `python -m graphids.sweep --multirun 'training.lr=interval(1e-4, 1e-2)' stage=autoencoder model=vgae_large dataset=hcrl_sa`

**Pros:** Full sweeper plugin compatibility (search space, pruning, storage, callbacks). Zero custom code for the HPO loop.
**Cons:** Second entry point (sweep.py vs cli.py). `@hydra.main` takes over `sys.argv` and working directory management. Need to suppress Hydra's output directory creation.
**Risk:** `execute_stage()` was designed for single invocation. If `@hydra.main` calls it N times in the same process, need to verify no shared state (CUDA contexts, structlog bindings, etc.).

**Evidence:** Hydra docs explicitly state the Compose API (which we use in `cli.py`) **forfeits multirun support**: "Avoid using the Compose API in cases where @hydra.main() can be used, as doing so forfeits many of the benefits of Hydra such as Tab completion, **Multirun**, Working directory management, and Logging management." ([source: hydra.cc/docs/advanced/compose_api](https://hydra.cc/docs/advanced/compose_api)). Since the optuna sweeper IS a multirun plugin, `@hydra.main` is required.

### ~~Approach B: Programmatic sweeper API~~ (RULED OUT)

~~Use the sweeper's internals programmatically (without `@hydra.main`).~~

**Ruled out:** No stable programmatic API exists for the sweeper. The only extension point is `custom_search_space` which is marked **EXPERIMENTAL** in the docs and requires `@hydra.main` anyway. There is no way to invoke the sweeper from the Compose API.

### Approach C: Keep Optuna directly, slim down custom code (FALLBACK)

Don't use the sweeper plugin at all. Instead:
1. Move search space YAML into Hydra config groups (`conf/sweep/vgae.yaml`, etc.)
2. Use `optuna.distributions` to load from resolved config
3. Keep `run_sweep()` but slim it down — remove search space parsing, inline `build_cli_cmd()`

**Pros:** Minimal change. No new dependency. No `@hydra.main` compatibility question. Works with existing Compose API.
**Cons:** Still ~150 lines of custom sweep code. Doesn't achieve the "zero custom HPO" goal.
**Savings:** ~100 lines (search space loading + build_cli_cmd), not the full 374.
**When to use:** If the spike reveals `@hydra.main` cannot coexist with our Compose API setup, or if `execute_stage()` has shared-state issues when called multiple times in one process.

---

## Search space translation

Current custom YAML format → hydra-optuna-sweeper format:

### Current (config/search_spaces/vgae.yaml)
```yaml
training.lr:
  type: loguniform
  low: 1.0e-4
  high: 1.0e-2
vgae.latent_dim:
  type: choice
  values: [16, 32, 48, 64]
vgae.dropout:
  type: uniform
  low: 0.05
  high: 0.4
```

### hydra-optuna-sweeper equivalent (in config.yaml or sweep config group)
```yaml
hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
    direction: minimize
    n_trials: 20
    storage: sqlite:///.cache/kd-gat/optuna_sweeps.db
    study_name_source: auto  # or explicit
    params:
      training.lr:
        type: float
        low: 1.0e-4
        high: 1.0e-2
        log: true
      vgae.latent_dim:
        type: categorical
        choices: [16, 32, 48, 64]
      vgae.dropout:
        type: float
        low: 0.05
        high: 0.4
```

Translation is mechanical: `loguniform` → `float + log: true`, `uniform` → `float`, `choice` → `categorical`.

---

## subprocess_utils.py (72 lines) — why it goes away

`build_cli_cmd()` exists solely because `optuna_sweep.py` needs to construct CLI commands for subprocess dispatch. With hydra-optuna-sweeper:

- **Approach A**: sweeper calls the task function directly (in-process or via Hydra's launcher). No CLI string building.
- **Approach B**: same — programmatic API calls task function.
- **Approach C**: even here, `build_cli_cmd()` could be inlined into the slimmed `run_sweep()`.

No other code imports `subprocess_utils`. It's a single-consumer module.

---

## Warm-start with hydra-optuna-sweeper

Two options, both using Optuna's native APIs (not sweeper-specific):

**Option 1: Pre-populate via SQLite storage.** Since the sweeper supports `storage: sqlite:///path` and `load_if_exists`, we can enqueue warm-start trials into the study BEFORE launching the sweep. A standalone script or `custom_search_space` callback (EXPERIMENTAL feature per docs) calls `study.enqueue_trial(prior_params)`. This is exactly what `_enqueue_warm_start()` does today.

**Option 2: `custom_search_space` callback.** The docs show this as an EXPERIMENTAL feature ([hydra.cc/docs/plugins/optuna_sweeper](https://hydra.cc/docs/plugins/optuna_sweeper)):
```python
def configure(cfg: DictConfig, trial: Trial) -> None:
    # Can call trial.suggest_* with custom logic
    trial.suggest_float("z", cfg.x - cfg.max_z, cfg.x + cfg.max_z)
```
This could load prior results and conditionally enqueue. But the EXPERIMENTAL label is a concern.

**Recommendation:** Option 1 — pre-populate the SQLite study. It's Optuna-native, not sweeper-specific, and decoupled from the sweep run.

---

## Pipeline sweep orchestration (what moves, not what disappears)

`run_sweep_pipeline()` is sequential orchestration: sweep stage A → train best A → sweep stage B → train best B → sweep stage C → train best C → evaluate.

This is NOT HPO logic — it's pipeline DAG logic that happens to use HPO as a step. With `run_dag()` + submitit from the stage-executor work:

```python
# Pseudocode: sweep pipeline becomes a DAG
for stage in ["autoencoder", "curriculum", "fusion"]:
    sweep_job = executor.submit(run_sweep_via_hydra, stage, dataset, scale)
    sweep_job.result()  # block — next stage depends on this
    train_job = executor.submit(execute_stage, best_cfg, stage)
    train_job.result()
eval_job = executor.submit(execute_stage, cfg, "evaluation")
```

Multi-seed is a submitit array job:
```python
jobs = executor.map_array(execute_stage, [cfg_seed42, cfg_seed123, cfg_seed456], ["autoencoder"] * 3)
```

---

## Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Compose API incompatibility | Medium | High | Spike: test @hydra.main wrapper with our config groups |
| CUDA isolation lost (in-process trials) | Low | High | Verify sweeper uses subprocess launcher or add `hydra/launcher=submitit_local` |
| Warm-start regression | Low | Medium | Test enqueue_trial with SQLite storage |
| Pipeline sweep complexity moves elsewhere | Certain | Low | It's already orchestration code — natural home is run_dag() |

---

## Spike questions (for .research.md)

1. **Does `@hydra.main` work alongside our `compose_config()` in the same package?** Can we have `cli.py` (compose API) + `sweep.py` (@hydra.main) coexisting?
2. **How does the sweeper handle subprocess isolation?** Does it fork, spawn, or use Hydra launchers? We need CUDA context isolation.
3. **Can we use `hydra-submitit-launcher` for local subprocess dispatch?** (submitit's `LocalExecutor` for single-node HPO, `SlurmExecutor` for distributed)
4. **What's the sweeper's return value protocol?** Does it read from the task function return, a file, or Hydra's `HydraConfig`?
5. **SQLite storage path**: can we configure it to use our existing `.cache/kd-gat/optuna_sweeps.db`?
6. **Pruning**: does the sweeper support `MedianPruner`? Our current code uses it.

---

## Preliminary recommendation

**Approach A** (thin @hydra.main wrapper) looks most promising:
- Full plugin compatibility, zero custom HPO code
- Clean separation: `cli.py` for single runs, `sweep.py` for HPO
- `subprocess_utils.py` deleted entirely
- Pipeline sweep orchestration moves to `run_dag()` where it belongs

But the Compose API compatibility is the key unknown. The `.research.md` should spike this first — if `@hydra.main` can coexist with our Compose API setup, Approach A is the clear winner. If not, Approach C is the safe fallback (still saves ~100 lines).

---

## File impact (Approach A, if spike succeeds)

| Action | File | Lines |
|--------|------|------:|
| **Delete** | `pipeline/orchestration/optuna_sweep.py` | -302 |
| **Delete** | `pipeline/subprocess_utils.py` | -72 |
| **Delete** | `config/search_spaces/*.yaml` (3 files) | -57 |
| **Create** | `graphids/sweep.py` (@hydra.main entry point) | +15 |
| **Create** | `config/conf/sweep/vgae.yaml` | +15 |
| **Create** | `config/conf/sweep/gat.yaml` | +15 |
| **Create** | `config/conf/sweep/dqn.yaml` | +12 |
| **Modify** | `config/conf/config.yaml` (sweeper defaults) | +10 |
| **Modify** | `cli.py` (remove sweep subcommand or redirect) | -20 |
| **Modify** | `pyproject.toml` (add hydra-optuna-sweeper dep) | +1 |
| **Net** | | **~-383** |
