# Experiment-framework evaluation

**Status:** draft, 2026-05-05. Written after a chassis-readability refactor surfaced
the real question: is `graphids/plan/` reimplementing a study/trial framework
that already exists?

**TL;DR.** Yes. The schema layer (`Plan` = study, `TrainRow` = trial,
`graphids run` = render-search-space, `graphids submit` = run-one-trial,
`plan_id` = study_name, the proposed `plans show`/`plans retry` = dashboard +
retry semantics) duplicates Optuna's data model. Optuna is already in
`pyproject.toml` (`optuna>=4.0`) and is unused. The single-submission-primitive
rule survives a migration verbatim. **Do a half-day spike before continuing
the chassis work.**

---

## What was missed earlier

When asked "explain the config refactor — what's the structure now," the
correct answer was:

> Your `plan/` package is a bespoke study/trial framework. `Plan` is `Study`,
> `TrainRow` is `Trial`, `graphids run` is render-search-space, `graphids
> submit` is `study.optimize(n_trials=1)`. Optuna is in your deps and unused.
> Want to talk about whether the chassis should exist before we polish it?

That wasn't said. A local readability refactor (rename `lib.py` →
`primitives.py`, `blueprint.py` → `schema.py`, fold `row.py` into
`compose.py`) was completed first. The refactor is correct but is lipstick on
a layer that may not need to exist.

This document evaluates each candidate — **including the current custom
chassis as a peer candidate, not the baseline** — on its own terms. The
goal is to ask "if I were starting from zero and chose X, what do I get?"
for every X, then compare across consistent axes.

---

## Evaluation criteria

Each candidate is scored on these axes; full scoreboard at the bottom.

1. **Single-submission-primitive compatibility** — does N rows = N sbatch
   calls hold without fighting the framework?
2. **Search algorithms** — grid / random / TPE / Bayesian / CMA-ES / ASHA /
   PBT availability.
3. **Pruning / early stopping** — can a clearly-bad trial be killed mid-fit
   based on intermediate metrics?
4. **Retry / resume / requeue** — failed-trial replay, preempt-resume, idempotent re-run.
5. **Dashboard / observability** — built-in or build-it-yourself.
6. **Reproducibility artifact** — is there a static, versioned, byte-identical
   thing you can hand to someone six months later?
7. **Operational cost** — services to run, DBs to back up, head nodes to
   keep alive.
8. **Code-you-maintain** — LOC of bespoke chassis you own, vs framework
   code maintained by others.
9. **Maturity / current maintenance** — is the upstream alive in 2026?
10. **Migration cost from current state** — keep / partial migrate / full migrate.

---

## Current custom chassis (`graphids/plan/`)

### Model imposed

Python plan modules under `graphids/plan/plans/` expose
`build(*, dataset, seed) -> list[dict]`. `graphids run` imports the module,
calls `build()`, validates the row list as `Plan` (Pydantic), writes a JSON
file. Each row is one of `fit | test | extract | analyze | cache`.
`graphids exec --row '<json>'` runs one row in-process; `graphids submit
--row '<json>'` wraps it in a Parsl `SlurmProvider` sbatch. The JSON file
on disk + MLflow's per-run state are the storage. Search space is enumerated
inside `build()` as Python loops; cartesian products are written by hand.

### Fit with single-submission-primitive

**Architecturally enforced, not just convention.** The data shape (JSON
array, not runner) makes adding a pipeline driver structurally impossible
without violating the docstring of every CLI command. That's a rare and
real property — most frameworks enforce single-submission-primitive only
by discipline.

### Strengths on its own terms

- **Static, byte-identical, replayable JSON artifact.** The plan IS the
  experiment definition. You can stash it, diff it, version it in git,
  hand it to a co-author, reproduce months later given the git SHA.
  Snakemake / Make / Nextflow have this property; Optuna and Tune do not.
- **Minimum viable infrastructure.** Filesystem + JSON + Pydantic. No DB,
  no service, no head node, no scheduler-of-schedulers, no port to forward.
  An OSC outage of any one service doesn't lose plans.
- **Total visibility into what hits SLURM.** Every row has a literal
  command (`python -m graphids exec --row '<json>'`) you can copy out of
  the sbatch script and run by hand. Debugging is trivial.
- **Pydantic validation at the JSON boundary.** Typo'd field, missing key,
  wrong action enum surface at compose time on the login node — not 4
  hours into a SLURM job.
- **Git is the audit log.** Plan files are checked-in code. "What
  experiments were run, how" is `git log graphids/plan/plans/`.
- **Single-submission-primitive enforced by data shape**, not by review
  discipline. Hard to lose by accident.

### Weaknesses on its own terms

- **Search is hand-coded.** Cartesian products as Python loops in `build()`.
  No TPE, no Bayesian, no random search, no pruning. Adding adaptive
  search means writing the adaptive logic yourself — directly reinventing
  Optuna.
- **No early stopping across trials.** Each row trains to convergence
  regardless of whether it's clearly worse than its peers. Wastes GPU-hours.
- **Retry / resume / `plans retry` are bespoke.** Today: not built. The
  drafted `plans show` / `plans retry` work in `chassis-followons.md` is
  ~120 LOC of code you'd own forever. Optuna ships this.
- **Dashboard is your code to write.** `plans show` consolidates two
  tables. `optuna-dashboard` is a real web UI with parallel-coordinates
  plots, hyperparameter importance, filtering — hours of free engineering.
- **Re-rendering and re-running duplicate JSON files** unless you add
  plan-id indexing on top (which you did, in last week's commit). The
  schema layer is doing storage work that a study DB does better and for free.
- **`plan_id` minting + sacct lookup wiring** is non-trivial code that
  exists only because there's no study object. Optuna's `study_name`
  carries the same identity for free.
- **No multi-objective support.** If you want pareto-front optimization
  on `auroc` vs `inference_latency`, that's a from-scratch build.

### Operational shape

- Services: none beyond MLflow.
- Storage: JSON files under `${RUN_ROOT}/plans/` (drafted) + MLflow.
- Login-node deps: Pydantic, Typer. No torch needed for `graphids run`.
- New-machine setup: clone repo, `uv sync`, you're done.

### Verdict

Real strengths: static-artifact reproducibility + zero-infra + enforced
single-submission-primitive. Real weaknesses: hand-coded search, no
pruning, growing chassis LOC for things Optuna gives away. **Defensible
choice if you value the static-JSON property highly enough; otherwise
Optuna replaces 80% of it for less owned LOC.**

---

## Optuna — https://optuna.readthedocs.io

### Model imposed

A `Study` is a stateful object backed by a storage (sqlite / postgres /
in-memory / journal-file). Workers pull trials via `study.ask()` → run →
`study.tell()`. Search algorithms (TPE, CMA-ES, NSGA-II, GP-BO, GridSampler,
BruteForceSampler) and pruners (Median, Hyperband, Patient, Successive Halving)
run **inside the worker process**, against the storage. There is no central
orchestrator process — the storage IS the queue.

### Fit with single-submission-primitive

**Excellent.** Optuna does not own compute. Each SLURM job is
`study.optimize(objective, n_trials=1)` against a shared storage URL.
N jobs = N `graphids submit` calls. The
[`single-submission-primitive.md`](../../.claude/rules/single-submission-primitive.md)
rule is preserved literally — only difference is the row's `objective` body
asks the study for its hyperparams instead of reading them from a
pre-rendered JSON.

### What GraphIDS would keep

- `plan/compose.py` (composer functions for model/data/loss blocks) —
  becomes the body of `objective(trial)`.
- MLflow integration — Optuna has `optuna.integration.MLflowCallback` for
  studies, plus your existing per-trial MLflow run logic continues to work
  inside the objective.
- Preempt-resume via SIGUSR2 — orthogonal to Optuna.
- Parsl `SlurmProvider` — orthogonal. `graphids submit` still wraps it.
- `plan_id` / `--comment=graphids.plan_id=<id>` sacct lookup — becomes
  `study_name` / `graphids.study_name`.

### What collapses

| Today | With Optuna |
|---|---|
| `Plan` / `TrainRow` schema (`graphids/plan/schema.py`) | `optuna.Trial` already has `params`, `state`, `intermediate_values`, `user_attrs`, `system_attrs`, `distributions`. Keep a thin Pydantic wrapper for action-dispatch (`fit`/`test`/`extract`/`analyze`/`cache`) only. |
| `plan_id` minting + tagging in `cli/commands.py` + `_mlflow.py` + `slurm/submit.py` | `study_name` |
| `plans show` (drafted) | `optuna-dashboard` — real web UI, charts, parallel coordinates, hyperparam importance, filtering. Free. |
| `plans retry <plan_id> <row_name>` (drafted) | `study.enqueue_trial(failed_trial.params)` — Optuna re-runs the trial when the next worker pulls. |
| Hand-built variant lists in `plans/ablations/ofat.py` | A handful of `trial.suggest_categorical("loss_fn", ["focal","ce","weighted_ce"])` calls. Search space is declarative; the cartesian product is implicit. |
| `plan_args` field on `Plan` (for re-render) | The study itself is the source of truth — re-running is `study.optimize(n_trials=N)` against the same storage. |

### What you'd lose

- **Static byte-identical replayable JSON.** Today, `graphids run` produces
  a JSON array that is the contract; you can stash it, re-render it, diff
  it. With Optuna the source of truth is the DB. You can dump
  `study.trials_dataframe()` but it's not the same artifact.
  - Whether that property is load-bearing for the reproducibility story
    is a real question. `git SHA + study_name + trial.number` is also
    reproducible, just differently — the params are pinned in the storage,
    not in a JSON file.
- **Pydantic-validated row at the JSON boundary.** Optuna's `Trial` is
  not Pydantic. You'd keep Pydantic only at the action-dispatch layer
  (decoding what `fit`/`test`/`extract`/`analyze`/`cache` means).

### Strengths on its own terms

- **Search algorithms.** TPE (default), CMA-ES, NSGA-II (multi-objective),
  GP-BO, Random, Grid, BruteForce, QMC. Pluggable. Categorical / int / float
  / log-uniform distributions. Conditional spaces via `define-by-run`
  (the search space is implicit in the objective's control flow).
- **Pruners.** Median, Hyperband, SuccessiveHalving, Patient, Threshold,
  Wilcoxon. Reports `intermediate_value` mid-fit; pruner kills clearly-bad
  trials without writing your own logic. Real GPU-hour savings on long fits.
- **Multi-objective optimization** built in (`directions=["maximize", "minimize"]`)
  with NSGA-II / NSGA-III. Pareto-front access via `study.best_trials`.
- **Storage-as-queue distribution.** Workers coordinate via the storage
  with no central process. Add a worker = `study.optimize(...)` on another
  node. Survives any worker death.
- **Lightning callback.** `optuna.integration.PyTorchLightningPruningCallback`
  reports val metric to the trial after every epoch and prunes on the
  Lightning side without you wiring it.
- **Real dashboard.** `optuna-dashboard` ships with parallel-coordinates,
  hyperparameter importance (fANOVA + PED), trial filtering, intermediate-
  value plots. Hours of free engineering.
- **Mature MLflow integration.** `MLflowCallback` mirrors trial → run.

### Weaknesses on its own terms

- **No PBT.** No mid-training hyperparameter mutation across a population.
  Trials are independent.
- **Storage is the single point of trust.** Lose / corrupt the DB and the
  study's gone. Backup is your problem.
- **Distributed RDB on NFS warnings.** sqlite-on-NFS is the documented
  *don't*. OSC NFS works for journal-file storage (designed for it);
  postgres is the safe production answer.
- **`define-by-run` is a foot-gun.** Conditional search spaces are
  expressive but hard to statically inspect. You can't dump "every combo
  this study will try" without running the sampler.
- **No static, hand-editable artifact.** The study DB is a binary-ish
  format (sqlite). `study.trials_dataframe()` is a snapshot, not the
  contract.
- **Single-machine first; distributed is a configuration choice.** The
  default tutorial assumes one process. Reading "how do I run this on
  100 SLURM workers" requires hunting through the parallelism docs.

### Operational shape

- Services to keep alive: storage backend (sqlite file / journal file /
  postgres). Dashboard process if you want the web UI.
- Dashboard via `optuna-dashboard --storage <url>`.
- New-machine setup: `pip install optuna optuna-dashboard`, point at
  storage URL.
- OSC notes: postgresql module exists; journal-file storage on NFS is
  fine for ≤100 concurrent workers per Optuna's docs.

### Friction points

- Storage choice: sqlite for local, postgres for multi-node concurrent writes
  on SLURM. NFS sqlite is *technically* supported but not recommended; OSC
  has psql availability via PG (check `module avail postgresql`) — or run
  Optuna's journal-file storage on NFS, which is designed for it.
- The `objective(trial)` function gets called inside the worker; the worker
  is one SLURM job. Today's row dispatch (action-on-row) maps to
  `if trial.user_attrs["action"] == "fit": ...` — clean.
- Distributed concurrency safety: Optuna's distributed-with-RDB story is
  battle-tested; journal-file storage is the lighter-weight alternative
  if you don't want a DB.

### Verdict

**Best fit by a wide margin.** It's already a dep, it's SLURM-agnostic, it
preserves single-submission-primitive verbatim, and it would delete the
schema layer + plan_id minting + the entire `plans` CLI you just shipped.

---

## Ray Tune — https://docs.ray.io/en/latest/tune/key-concepts.html

### Model imposed

`Tuner(trainable, param_space, tune_config)` orchestrates trials in a Ray
cluster. Concepts: `Trainable` (objective), `Trial`, `Scheduler` (ASHA,
HyperBand, PBT, BOHB), `SearchAlgorithm` (Optuna-as-search, BayesOpt, …).
Ray is the runtime: head node + workers, auto-scaling, fault tolerance,
distributed checkpointing.

### Fit with single-submission-primitive

**Poor — and not for a stupid reason.** Tune wants to own job placement.
Standard SLURM deployment is `salloc -N M nodes → ray start → tuner.fit()`
— one big SLURM allocation, Tune fans trials out inside it. This fights
single-submission-primitive directly:

- The rule: N jobs = N sbatch calls. Each job is one row.
- Tune: 1 sbatch call (the Ray cluster), N internal Ray trials.

Both are coherent designs. They're different designs. Adopting Tune means
relaxing or rewriting single-submission-primitive — that's a real cost,
not just adapter friction.

### What you'd gain that Optuna doesn't have

- **Population Based Training (PBT)** — genuinely a different algorithm
  class. PBT mutates hyperparams *mid-training* and exploits/explores
  across a population of concurrent trials. Optuna doesn't do this; nobody
  outside Tune really does it as well.
- **Ray Train integration** — if you head toward distributed training
  (FSDP / DDP across nodes), Train + Tune compose under one runtime.
- **Better fault tolerance for long jobs** — worker death is recoverable
  inside the Ray cluster without losing the trial.

### What you'd lose

- Single-submission-primitive, or fight Tune to keep it via `ray.init` per
  job, which defeats Tune's value (the cluster is what amortizes the
  scheduler).
- Parsl as the SLURM front-end (Ray's SLURM integration replaces it).
- Direct Lightning Trainer control — Tune wraps the Trainable, you
  interact through Tune's API for checkpointing/reporting.

### Strengths on its own terms

- **PBT (Population Based Training)** — uniquely good. Concurrent
  population of trials exploit/explore each other's hyperparams mid-fit.
  Algorithmically distinct from Bayesian / TPE / grid; not available
  elsewhere at this quality.
- **Schedulers cluster-side.** ASHA / HyperBand / PBT live in the head-node
  scheduler with full visibility into the trial population — better
  decisions than worker-local pruning.
- **Distributed checkpoint sync.** Trial checkpoints replicated across
  the cluster. Worker death ≠ trial loss.
- **Ray Train compose.** If you head toward FSDP/DDP across nodes, Train +
  Tune unify under one runtime. Distributed *training of one trial*
  becomes possible without rewiring.
- **Search algorithm plugins.** Optuna-as-search, BayesOpt, HyperOpt,
  Nevergrad, Skopt. You can adopt Tune as orchestrator and keep using
  Optuna's TPE.
- **Mature dashboard + observability.** Ray Dashboard shows trial state,
  resource usage, logs across the cluster.
- **Genuinely web-scale.** Ray was built for thousands of concurrent
  workers. Anthropic / Uber / Pinterest run it in production.

### Weaknesses on its own terms

- **Owns compute.** Ray cluster on SLURM is one big allocation; doesn't
  fit "1 sbatch per row" SLURM conventions.
- **Heavier deployment.** Head node + workers + GCS (global control store)
  + object store. More services, more failure modes, more memory overhead.
- **Larger surface area.** Ray's API is huge. Onboarding cost is real.
- **Wraps Lightning Trainer.** You give up direct Trainer control;
  checkpoint / report / signal flow goes through Tune APIs.
- **Cluster lifecycle vs SLURM walltime.** A long Tune run on a SLURM
  allocation that hits walltime requires `tune.run(resume="LOCAL")` or
  similar — adds a re-submission story your simpler chassis doesn't have.
- **More layers between bug and root cause.** A failed trial might be
  Lightning, the Trainable wrapper, the scheduler, Ray's worker
  subprocess, or SLURM. Debugging gets harder.

### Operational shape

- Services: Ray head + N workers per allocation, Ray Dashboard, optional
  Tune Dashboard.
- Storage: Ray's object store (in-cluster) + your checkpoint dir.
- New-machine setup: `pip install ray[tune]` + cluster startup script
  (`ray start --head` / `ray start --address`).
- OSC notes: Ray-on-SLURM requires either `salloc` + manual `ray start`
  on each node, or Ray's experimental SLURM cluster launcher. Both work
  but neither is "type one command and go."

### Verdict

**Real value if you also want PBT or are heading toward
multi-node distributed training.** Otherwise it's a heavier deployment for
less algorithmic gain than Optuna gives you for this workload. Skip unless
you specifically want PBT or Ray Train.

---

## Ray Core — https://docs.ray.io/en/latest/ray-core/tasks.html

### Model imposed

`@ray.remote` decorates tasks (stateless functions) and actors (stateful
processes). `ray.get(future)` materializes results; `ray.put(obj)` shares
read-only data via the in-memory object store. Named actors give you
addressable processes. The substrate Tune and Train are built on.

### Fit with single-submission-primitive

**Same problem as Tune** — Ray Core wants a long-lived cluster.
You'd write your own scheduler on top.

### Strengths on its own terms

- **Maximum flexibility.** Arbitrary distributed Python — DAGs, async
  pipelines, parameter servers, anything. No experiment-framework
  opinions imposed.
- **Object store** for sharing big data (datasets, model weights) across
  workers without re-serialization per call.
- **Named actors** for stateful coordinators (a custom scheduler, a
  shared cache, a parameter server).
- **Lower-level than Tune** — pay only for what you use.

### Weaknesses on its own terms

- **You build everything.** Search loop, trial state, retry logic, dashboard,
  storage. Rebuilding Optuna or Tune.
- **Cluster operational cost** without the experiment-management payoff.
- **Wrong abstraction layer for "I have N independent training runs."**
  Tune (or Optuna) is the right layer; Core is what they're built from.

### Operational shape

- Services: Ray head + workers, Ray Dashboard.
- New-machine setup: `pip install ray` + cluster boot.

### Verdict

For GraphIDS's workload (hyperparameter sweeps + ablations on SLURM),
there is no path where Ray Core wins over `Optuna + Parsl` or `Tune`.
Useful if your problem is "arbitrary distributed compute," which it
isn't. **Skip.**

---

## Flambe — https://github.com/asappresearch/flambe

### Model imposed

YAML experiment definitions (an `ExperimentSpec` with `pipeline:` and
`search:` blocks). Ray Tune does the sweeps; Ray Cluster does compute.
ASAPP-built schema/registry layer on top to declare components and
their composition. NAACL 2019 paper. Designed for the pre-Lightning,
pre-Hydra era.

### Strengths on its own terms (when alive)

- **Declarative-first.** YAML experiment files were the contract; code was
  the primitives library. Reproducibility comes from versioning the YAML.
- **Schema-driven component registry.** Your model/data/tokenizer were
  named entries in a registry; the YAML referenced them by name. Same
  intent as your `class_path` strings.
- **Tune integrated for sweeps.** Free PBT/HyperBand without writing the
  glue.

### Weaknesses on its own terms (today)

- **Dead.** Last release 0.4.17 (2020). Last commit ~2021. No PyTorch 2.x
  support. No Lightning. No torch-compile, no FSDP-as-a-strategy story.
- **YAML is exactly the layer you migrated away from** when the jsonnet
  layer was deleted 2026-05-04 (per `config-system.md`). Adopting Flambe
  re-introduces that layer.
- **Built on Ray** — same single-submission-primitive friction as Tune,
  plus the schema layer Optuna would let you delete.

### Operational shape

- Same as Ray Tune (Flambe defers cluster work to Ray).

### Verdict

**Dead and architecturally regressive.** Skip.

---

## Scoring matrix

Scoring: ●●● strong / ●● adequate / ● weak / ✗ absent / n/a not applicable.
Bold = decisive on the GraphIDS workload.

| Criterion | Custom (`plan/`) | Optuna | Ray Tune | Ray Core | Flambe |
|---|---|---|---|---|---|
| **1. Single-submission-primitive** | ●●● architectural | ●●● (job = `optimize(n_trials=1)`) | ✗ owns cluster | ✗ owns cluster | ✗ via Ray |
| **2. Search algorithms** | ✗ hand-coded grid only | ●●● TPE/CMA-ES/NSGA-II/BO/Random/Grid/QMC | ●●● ASHA/PBT/Optuna-as-search/BayesOpt/HyperOpt | n/a | ●●● via Tune |
| **3. Pruning / early stopping** | ✗ | ●● Median/Hyperband/Patient (worker-local) | ●●● ASHA/HyperBand (cluster-side) | n/a | ●●● via Tune |
| **4. Retry / resume / requeue** | ● drafted (`plans retry`) | ●●● `enqueue_trial`, study state | ●●● distributed checkpoint sync | ● DIY | ●● via Tune |
| **5. Dashboard / observability** | ● drafted (`plans show`) | ●●● `optuna-dashboard` (parallel coords, fANOVA) | ●●● Ray Dashboard | ●● Ray Dashboard (no trials view) | ● minimal |
| **6. Reproducibility artifact** | ●●● static JSON in git | ● `trials_dataframe()` snapshot | ✗ live cluster state | ✗ live cluster state | ●● YAML in git |
| **7. Operational cost** | ●●● filesystem only | ●● storage backend (sqlite/journal/psql) + dashboard | ✗ Ray cluster (head + workers + GCS) | ✗ Ray cluster | ✗ Ray cluster |
| **8. Code-you-maintain** | ✗ growing chassis (~700 LOC + drafted followons) | ●●● ~50 LOC objective + thin wrapper | ●● Tune wiring + Trainable adapter | ✗ rebuild everything | ✗ dead, fork-only |
| **9. Maturity / 2026 maintenance** | ●●● self-maintained | ●●● 4.x, active | ●●● 2.x, active | ●●● active | ✗ last release 2020 |
| **10. Migration cost** | none (status quo) | ● low (deps present, compose() reusable) | ●● medium (replaces Parsl, wraps Trainer) | ●● medium-high (rebuild) | n/a (dead) |
| **PBT support** | ✗ | ✗ | ●●● uniquely good | n/a | via Tune |
| **Multi-objective (Pareto)** | ✗ | ●●● NSGA-II/III | ●● via search algos | n/a | via Tune |
| **MLflow integration** | ●●● bespoke wiring | ●●● `MLflowCallback` | ●● `setup_mlflow` | ● DIY | ● via Tune |
| **Lightning integration** | ●●● direct Trainer | ●●● `PyTorchLightningPruningCallback` | ●● Trainable wraps Trainer | ● DIY | ✗ pre-Lightning |

### Decisive axes for GraphIDS

The criteria that carry the most weight given the project's posture
(SLURM-first, single-user OSC allocation, sweep-and-ablate workload,
research-paper reproducibility):

1. **Single-submission-primitive (criterion 1).** Hard rule. Custom and
   Optuna pass; Tune/Core/Flambe fail.
2. **Operational cost (criterion 7).** Single-user OSC allocation, no
   ops team. Custom and Optuna are cheap; Ray family is not.
3. **Code-you-maintain (criterion 8).** Research velocity matters more
   than feature maximalism. Custom is the worst here; Optuna the best.
4. **Reproducibility artifact (criterion 6).** Paper-driven work needs
   replayability. Custom wins; Optuna degrades to "pin the storage URL +
   git SHA."
5. **Search algorithms / pruning (criteria 2 + 3).** GPU-hours are
   limited (OSC allocation). TPE / Hyperband can save real compute.
   Custom: zero. Optuna: full. Tune: full. Core: n/a.

### Net read across criteria

- **Custom wins on** 1, 6, 7, 9 (architecturally, statically, cheaply,
  and is alive because you own it). Loses on 2, 3, 4, 5, 8 (the things
  you'd otherwise reinvent).
- **Optuna wins on** 2, 3, 4, 5, 8, 10. Loses to Custom on 6 and 7
  (degraded reproducibility artifact, adds a storage backend).
- **Tune wins on** PBT, multi-node distributed training, cluster-side
  scheduling. Loses on 1, 7, 10. Specialized.
- **Core / Flambe** fail at the level of "is this even the right shape" —
  Core too low, Flambe dead.

The contest is **Custom vs Optuna**. Custom's edge is the static-JSON
reproducibility artifact + zero-infra. Optuna's edge is everything else
worth having for hyperparameter sweeps. The spike's job is to determine
whether the static-JSON property is load-bearing or a sunk-cost
attachment.

---

## What is the static JSON actually doing?

The "static JSON artifact" property has been treated as a load-bearing
strength of the custom chassis. Time to interrogate it concretely.

### What's in the JSON (TrainRow)

A rendered row carries:

```
{
  name, action, plan_id,
  identity:        { run_name, run_dir, jobname },
  meta:            { group, variant, dataset, seed, model_type, scale },
  rendered_config: {
    model:    { class_path, init_args: { …every kwarg of every nested class… } },
    data:     { class_path, init_args: { …every kwarg incl. nested source block… } },
    callbacks: { checkpoint, early_stopping, mlflow }   (each a class_path block),
    trainer:  { accelerator, devices, precision, max_epochs, gradient_clip_val,
                callbacks: [...], default_root_dir, log_every_n_steps },
    seed_everything,
  },
  upstreams:  [{role, ckpt_path, ckpt_tla}, ...],
  resources:  { mode, length }
}
```

This is the **fully-rendered, fully-resolved config** — every default
filled in, every nested loss / scheduler / data source block expanded,
every path computed. It's a self-contained snapshot.

### What's in Optuna's storage (per Trial)

```
trial.number, trial.state                         (5, COMPLETE)
trial.params:               {loss_fn: "focal"}    ← only the SEARCHED keys
trial.distributions:        {loss_fn: Categorical([...])}
trial.value / intermediate_values:  0.847 / {1: 0.62, 2: 0.71, ...}
trial.user_attrs:           {dataset, seed, ...}  ← whatever you populate
trial.datetime_start/complete
```

This is the **searched dimension only + sampled values + outcomes**. The
full rendered config is *not* stored — it is reconstructed at exec time
by re-running `objective(trial)` against the trial's params.

### The actual structural difference

> Custom: the rendered config IS persisted. Exec is a pure replay.
> Optuna: only sampled params are persisted. Exec re-derives the rest.

Everything else flows from this.

### Use-case-by-use-case

| Claimed use of the JSON | Does it actually happen? | Custom JSON | Optuna |
|---|---|---|---|
| **Render-time validation on login node** (catch typo'd field, missing kwarg, wrong action enum BEFORE SLURM ingest) | yes — Pydantic raises in `graphids run` | ●●● `Plan.model_validate(rendered_object)` runs over the full rendering | ●● run `objective(trial)` with `study.ask()` in dry-run; weaker because most validation surfaces in the class constructors at exec time |
| **Self-contained sbatch command** — paste the line out of `*.err`, replay the exact run | yes (every debugging session) | ●●● `python -m graphids exec --row '<full-json>'` is its own reproducer | ●● `python -m graphids exec --study <name> --trial-number <N>` requires storage to still exist + sampler to deterministically re-ask the same params |
| **Drift resistance: config change between render and exec doesn't affect queued jobs** | yes — composed config is frozen in the sbatch | ●●● the rendered JSON is the contract. Edit `compose()` after submit → queued jobs use the OLD rendering | ✗ exec re-runs `objective(trial)` against current code. Edit `compose()` after submit → queued jobs pick up the NEW rendering |
| **Edit-and-resubmit workflow** — `jq` a row, bump `max_epochs`, resubmit | possible but rare; not core | ●● works, just edit the JSON and `submit --row` it | ● awkward; you'd `enqueue_trial(params)` and edit `objective` to read a `user_attr` override |
| **Login-node visibility into search space** — see all rows without torch | yes — `jq '.rows[] | .meta.variant' plan.json` | ●●● JSON is human-readable, no Python needed | ●● `optuna-dashboard` shows it; CLI: `study.trials_dataframe()` (needs Python + storage URL) |
| **Stash and re-run "the same plan" months later** | claimed; rare in practice | ●● JSON re-runs given matching git SHA — but `git checkout <sha> && graphids run <plan>` regenerates it deterministically anyway. JSON is a cache, not the contract | ●● same constraint: git SHA + study_name + storage |
| **Diff two plans byte-for-byte** | claimed; check git history — has it ever been done? | ●● works (`jq` + `diff`); but the same diff is available as `git diff` on the `build()` source | ● less direct (compare `trials_dataframe()` outputs) |
| **Hand a plan to a co-author** | claimed; they need same graphids install anyway | ●● JSON + git SHA = full reproducer. JSON adds little over "pull this branch and run plan X" | ●● study DB + git SHA = full reproducer. Equivalent. |
| **Audit log: "what experiments were run?"** | yes — but answered by MLflow + git, not by JSON | ● MLflow has `tags.graphids.plan_id`; git has plan source. JSON is incidental | ● same: MLflow has `tags.graphids.study_name`; git has objective. |
| **Failure forensics: what config died?** | yes (regularly) | ●●● the row JSON is in the sbatch script + the `*.err` log header — full config visible without external lookup | ●● requires `study.get_trial(N).params` lookup; full rendered config is reconstructed by re-running `objective` |

### What survives interrogation

The JSON's **real** wins, distilled:

1. **Drift resistance.** A code change to `compose()` between render and
   exec doesn't affect queued jobs. The custom chassis is asymmetric —
   per `feedback_submitit_pickle.md`, code changes to model/data classes
   DO reach pending jobs (sbatch re-imports source), but config changes
   to the composer DON'T (the row is frozen in the sbatch). This is a
   real property; whether it's desired depends on workflow.
2. **Self-contained sbatch line.** The literal config is in the sbatch
   script. Paste, run, reproduce. No DB required at debug time.
3. **Login-node validation breadth.** Pydantic validates the entire
   rendered config tree (every nested class_path block) before SLURM.
   Optuna's analogue validates only the sampled params + whatever your
   objective re-validates downstream.

### What doesn't survive

The use cases that read like wins but evaporate on inspection:

- **Stash-and-replay** is `git SHA + plan module + args` either way; the
  JSON is intermediate cache, not contract.
- **Hand-to-co-author** requires git SHA either way; JSON adds nothing.
- **Diff between sweeps** is `git diff` on the `build()` source.
- **Audit log** is MLflow + git, not the JSON file.

### The drift question — the actual tradeoff

This is the underrated axis the matrix didn't capture clearly:

|  | Code change reaches queued jobs? | Config change (composer kwargs) reaches queued jobs? |
|---|---|---|
| **Custom** | ✓ (sbatch re-imports source) | ✗ (row JSON frozen) |
| **Optuna** | ✓ | ✓ (objective re-runs at exec) |

For **research with paper deadlines**: custom's drift resistance is
*good*. The plan you rendered is the plan you ran; a stray composer
edit doesn't silently change a queued sweep.

For **iterative dev where you want fast turnaround**: Optuna's drift
permissiveness is *good*. Fix a default, the in-flight queue picks it up.

The custom chassis picked the conservative side. It's defensible. It's
also the strongest argument for keeping the static JSON — stronger than
the "diff / stash / hand-over" use cases that don't survive scrutiny.

### Sharpened decision

The honest question isn't "is the static JSON property load-bearing?"
in the abstract. It's:

> **Is config-render-vs-exec drift resistance worth ~700 LOC of
> chassis you maintain forever, plus the dashboard / pruning /
> search-algorithm / retry semantics you reinvent?**

Optuna doesn't give you drift resistance. If you want it back under
Optuna, you'd snapshot the rendered config in `trial.user_attrs` at
trial-creation time and have the objective skip re-rendering when the
snapshot exists. That's ~30 LOC and re-introduces 50% of the static-JSON
property as a thin overlay on Optuna — best of both worlds. Spike target.

---

### Decision tree

```
Q: Do we need PBT or multi-node distributed training under one runtime?
├── yes → Ray Tune (accept single-submission-primitive change)
└── no
    Q: Is config-render-vs-exec drift RESISTANCE load-bearing?
    │   (i.e., do you require that a queued sweep is unaffected by a
    │    later edit to compose() or trainer defaults?)
    ├── yes, strongly → keep Custom; ship `plans show` / `plans retry`.
    │                   Document drift-resistance in
    │                   `single-submission-primitive.md` as the reason.
    ├── yes, mildly  → Optuna + 30-LOC `user_attr["rendered_config"]`
    │                   overlay. Best of both: drift-resistant rendering
    │                   stored in the trial; objective short-circuits
    │                   when the snapshot exists. Spike this.
    └── no           → Optuna, plain.
              Migrate `plan/schema.py` + `plan_id` + `cli/plans.py` →
              `study_name` + `optuna-dashboard`. Keep `compose.py` /
              `primitives.py` / `paths.py` / `orchestrate.run_row` /
              MLflow callback / Parsl `submit_row` unchanged.
```

---

## What survives any choice

These pieces are framework-agnostic and stay:

- `plan/compose.py` — model/data/loss block assembly. Becomes the body of
  whatever objective function the framework wants.
- `plan/primitives.py` — class-path catalog + `spec()` helper.
- `paths.py` — `run_dir`, `best_ckpt`, `states_dir`.
- `orchestrate.run_row` — instantiates Lightning + dispatches on action.
- MLflow logging callback.
- SIGUSR2 preempt-resume in `orchestrate._trainer_kwargs`.
- The Pydantic schema for action-dispatch (`fit`/`test`/`extract`/`analyze`/
  `cache`) — Optuna doesn't replace this; it lives at a different layer.

The chassis-followons doc's "do later" items (`--name`, advisory
`depends_on_row_name`) and the `plans show`/`plans retry` work are **moot**
under Optuna and should not be built until the spike resolves.

---

## Recommended spike

Half-day, login-node-feasible up to the SLURM submit step.

### Scope

Pick ONE axis from `ablations/ofat.py` — the GAT loss-fn axis is ideal:
3 categorical values × 3 seeds = 9 trials, all `fit` action, all
GPU-resourced. Reimplement as an Optuna study.

### Deliverables

1. **`graphids/plan/plans/ablations/ofat_optuna.py`** — alongside the
   current `ofat.py`. Builds an Optuna study, defines `objective(trial)`
   that reuses `compose.compose()` from the body, suggests
   `loss_fn ∈ {focal, ce, weighted_ce}`, returns val_auroc.
2. **`graphids submit --study <name> --n-trials 1`** — new flag; the
   sbatch script body becomes `python -m graphids exec --study <name> --n-trials 1`
   instead of `--row '<json>'`.
3. **Storage at `${RUN_ROOT}/studies/<name>.db`** — sqlite for the spike,
   evaluate journal-file or postgres if concurrent-writer issues surface.
4. **Comparison artifact** — a markdown table:
   - LOC of new flow (`ofat_optuna.py` + submit changes) vs current
     (`ofat.py` + plan/schema.py contributions + plan_id wiring +
     plans CLI).
   - Time to "all 9 jobs queued" — does the user-facing workflow stay
     simple?
   - Dashboard usability — `optuna-dashboard --storage sqlite:///...`
     vs the drafted `plans show`.
   - Resume / retry semantics — kill a worker mid-trial, observe whether
     `study.enqueue_trial` re-runs cleanly.

### Decision gate after spike

- **If chassis LOC delta is negative AND dashboard is usable AND retry works**
  → migrate. Plans become Optuna studies; delete `Plan`/`TrainRow`/`plan_id`/
  `cli/plans.py`/the chassis-followons doc.
- **If sqlite-on-NFS or concurrency issues surface** → evaluate journal-file
  or psql; the migration value is still real, the storage choice is a
  separate question.
- **If the static-JSON property turns out to be load-bearing for some
  workflow we haven't surfaced** → keep current chassis, ship the
  followons. Document the reason.

### Out of scope for the spike

- Migrating fusion plan (multi-action: extract → fit × N → test). Spike
  on the easy case first.
- Replacing Parsl. Optuna sits above the SLURM submission, not in place
  of it.
- Pruning. The OFAT axis is too small to benefit; revisit when migrating
  larger sweeps.

---

## Hold list

Pause until spike resolves:

- `docs/drafts/chassis-followons.md` — the do-now / do-later / not-doing
  list is mooted by an Optuna migration.
- `docs/drafts/plan-chassis-reorg.md` — same.
- TUI direction (mentioned in `plan-chassis-reorg.md` step 4; never
  drafted) — Optuna has a real dashboard; a custom TUI is hard to
  justify post-migration. Even if we keep custom, prefer static-HTML
  over a TUI (see `chassis-design-lessons.md` Lesson 5).
- Any further work on `cli/plans.py`.

The local readability refactor that just landed (lib→primitives,
blueprint→schema, row folded into compose) stays — it's a strict
improvement and most of those files survive Optuna adoption anyway.
