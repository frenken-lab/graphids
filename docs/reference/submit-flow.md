# Submit flow — atomic and plan submission, end to end

> Status: **implemented** | Companions: `orchestration.md` (build/train/evaluate),
> `observability.md` (MLflow + OTel layout)

How a job goes from CLI invocation on the login node to a running training
process on a compute node, including preemption recovery and plan-level
dep threading. Files referenced are clickable in any IDE-like reader; line
numbers are not pinned, names are.

## Two entry shapes

```bash
# Atomic — one preset, one job. Most common interactive use.
python -m graphids submit <preset.jsonnet> --dataset X --seed N [opts]

# Plan — multi-node DAG. `run` RENDERS a bash artifact; it does not submit.
python -m graphids run  configs/plans/ofat.jsonnet --dataset X --seed N --cluster C > runme.sh
bash runme.sh                                                          # actually submits
python -m graphids status configs/plans/ofat.jsonnet --dataset X --seed N   # read-only MLflow
```

The plan is *data*; the bash script is the *action*. Each line in the
artifact is one ``graphids submit`` primitive call — the same atomic
entrypoint a user would invoke by hand. afterok deps thread via shell
variables (``JID_<NAME>=$(graphids submit ...)`` then
``--dep "$JID_<UPSTREAM>"`` downstream). `run` and `submit` compose;
they don't duplicate.

## Atomic: `graphids submit <preset>`

Login-node phase (until `sbatch` returns):

1. **`graphids/cli/submit.py:submit_cli`** parses Typer flags. Validates
   that `--depends-on` and `--ckpt-path` aren't combined (different
   semantics — see "Three ckpt flags" below). Builds `flag_tlas` from
   the flat shortcuts (`--dataset`, `--seed`, `--scale`, `--ckpt-tla`,
   `--lake-root`).
2. **`--depends-on` resolution** (optional). For each `<variant>:<seed>`
   spec, `graphids/slurm/dependencies.py:resolve_dependency` calls
   `_mlflow.latest_run(..., status="FINISHED")`, reads the upstream's
   `graphids.run_dir` tag, and injects `<role>_ckpt_path` TLAs (e.g.
   `vgae_ckpt_path`). The producer→consumer-TLA mapping lives in
   `DEPENDS_ON_TLA`.
3. **`--skip-if-finished` short-circuit** (optional). Infers
   `(group, variant)` from the preset path, calls
   `_mlflow.is_finished(...)`. If FINISHED, prints `0` and exits — the
   bash idiom `jid=$(graphids submit ...)` still works.
4. **`graphids/slurm/submit.py:submit()`** — the one library entrypoint.
   `ensure_env_loaded()` (python-dotenv) populates `os.environ` from
   `.env`. Looks up `_PROFILES[mode][cluster][length]` from
   `configs/resources/submit_profiles.json` (raw `submitit.AutoExecutor`
   kwargs — no parser layer). Applies `--mem-gb` / `--timeout-min`
   overrides; optional `--time-from-history` queries MLflow via
   `graphids.slurm.sizing.estimate_walltime_minutes` (group p95 × 1.5).
   `_align_cpus_to_mem` bumps `cpus_per_task` to satisfy the cluster's
   `mem_per_cpu_gb` ratio.
5. **Render the preset once, on the login node.** For preset payloads,
   `submit()` calls `config.jsonnet.render(preset, tla=...)` and stamps
   the rendered `trainer.default_root_dir` into the `_TrainingJob`'s
   `run_dir` field. The compute node never re-renders jsonnet during
   preemption recovery.
6. **Build the work unit.** Preset → `_TrainingJob(action, config, tlas,
   sets, ckpt_path, run_dir)` (pickle-safe dataclass). `--command` →
   `submitit.helpers.CommandFunction(["bash", "-c", cmd])`.
7. **Submit via submitit.** `submitit.AutoExecutor(folder=log_dir).submit(payload)`
   pickles the job, generates the sbatch script with
   `slurm_setup=["source scripts/slurm/_preamble.sh"]` so module-load and
   venv-activation run before submitit's `srun python -u -m
   submitit.core._submit {folder}`. The dependency string
   `afterok:<jid1>:<jid2>...` is set from `dep_jids`. Returns the jid.
8. **`submit()` returns** `int | None` — real jid for a successful
   submission, `None` for `--dry-run`. CLI prints `jid` (or `0` for
   skip/dry-run) to stdout for bash-capture.

Compute-node phase (after `_preamble.sh`):

9. **`srun python -u -m submitit.core._submit {folder}`** unpickles the
   payload and calls it.
10. **`_TrainingJob.__call__`** imports `graphids.cli.training.fit` (or
    `test`) and invokes it as a bare Python function — bypassing Typer's
    root callback. Provider/spawn/CPU-thread setup that the root callback
    would have done is duplicated in `_prepare()` for the compute-node
    path.
11. **`_prepare()`** wires the run: `init_providers` (OTel + W&B),
    `ensure_spawn` (CUDA-safe multiprocessing), `configure_cpu_threads`,
    `ensure_tracking_uri` (MLflow). `render(config, tla, set_overrides)`
    re-evaluates jsonnet on the compute node — this time for the
    canonical `ResolvedConfig`. Pydantic validates via
    `ResolvedConfig.from_rendered`. Writes `resolved.json` and
    `overrides.json` under `run_dir` for replay/debugging.
12. **`build(resolved)`** instantiates trainer / model / datamodule via
    `importlib` + `filter_kwargs`. See `orchestration.md`.
13. **`train(artifacts, resolved)`** opens an MLflow run via
    `_mlflow.start_training_run` (status-gated resume — see
    `observability.md`), runs `trainer.fit(...)`, and
    `MLflowTrainingCallback.on_fit_end` closes the run FINISHED.

Preemption recovery (5 min before walltime):

14. SLURM sends SIGUSR2 (USR2 because NCCL catches USR1; configured via
    `slurm_signal_delay_s=300` in the profile).
15. submitit's signal handler calls `_TrainingJob.checkpoint()`, which
    reads `{run_dir}/checkpoints/last.ckpt` if present and returns
    `submitit.helpers.DelayedSubmission(replace(self, ckpt_path=resume))`.
    submitit `afterany`-resubmits with that ckpt as the resume source.
    No manual loop, no jsonnet re-render — `run_dir` was stamped at submit
    time.

## Plan: `graphids run <plan.jsonnet>`

A *plan* is a jsonnet file declaring `{ nodes: [Node, ...] }`. The plan
itself is the topology source of truth — there is no parallel Python
declaration. `graphids run` is a **renderer**: it emits a bash script
composed of `graphids submit` invocations and exits. No SLURM contact.

1. **`graphids/cli/run.py:run_cli`** renders the plan via
   `config.jsonnet.render` (passing `dataset` + `seed` as TLAs) and parses
   to `tuple[Node, ...]` via `graphids/slurm/dag.py:parse_plan` (Pydantic
   `extra="forbid"` — typos die fast). `--variants A,B` filter walks
   transitive upstream deps via `filter_with_upstream`.
2. **`Node`** derives `group` / `variant` from the
   `<group>/<variant>.jsonnet` preset path; the plan only declares
   `preset:`. Off-convention paths must declare `group` / `variant`
   explicitly or fail validation.
3. **`graphids/slurm/run.py:render_plan_script`** toposorts via
   `graphlib` and emits one bash assignment per node:
   `JID_<NAME>=$(graphids submit ...)`. Preset lines bake `(dataset, seed,
   cluster)` and the node's mode / length / mem / timeout overrides;
   command lines emit `--command "..."`. Upstream deps are referenced as
   `--dep "$JID_<UPSTREAM>"`.
4. **`--skip-if-finished`** is appended to every *preset* line by default
   (`--force` opts out). Command-mode nodes don't get it — they have no
   `(group, variant)` for the MLflow lookup. When a preset is skipped at
   bash-execution time, `graphids submit` prints `0`; downstream
   `--dep "$JID_X"` becomes `--dep "0"`, which `submit()` filters out
   (`j > 0`) before composing the SLURM `afterok:` string.
5. **Header** carries the original CLI invocation, the node count, and
   a content-addressed `plan_hash=<8-hex>` derived from the toposorted
   `(name, deps, preset, command)` tuples — same inputs always produce
   the same hash.

Composition the user does:

```bash
graphids run plan.jsonnet --dataset X --seed N --cluster C > runme.sh
# review / edit / commit runme.sh
bash runme.sh
# every line is a graphids submit invocation; jids capture into shell vars
```

## Status: `graphids status <plan.jsonnet>`

Reuses `_load_plan` + `query_all` from the run path. `query_all` calls
`_mlflow.latest_run` once per non-command node (commands report `NA`).
Output is a `rich` table by default; `--format json` is the
machine-readable shape.

## Three ckpt flags — when to use which

| Flag | Semantics | Wired to |
|---|---|---|
| `--ckpt-tla` | Set the jsonnet `ckpt_path` TLA. | `flag_tlas` → `_TrainingJob.tlas` → consumed by the preset. |
| `--ckpt-path` | Resume the *current* preset — passthrough to `python -m graphids fit/test --ckpt-path`. | `_TrainingJob.ckpt_path` → `cli.training.fit(ckpt_path=...)`. |
| `--depends-on V[:S]` | MLflow lookup → inject upstream-teacher ckpt as TLA (e.g. `vgae_ckpt_path`). | `flag_tlas` via `dependencies.build_dependency_tlas`. Conflicts with `--ckpt-path`. |

In a plan, `--depends-on` semantics fall out of the topology — fusion
nodes depend on `extract-states`, which writes a tensor cache to
`paths.states_dir`. Direct upstream-ckpt injection isn't used in the
shipped OFAT plan.

## One MLflow query helper

Three callers want "latest row for `(dataset, [group], variant, seed,
phase[, status])`": `_mlflow.is_finished` (skip-if-finished),
`slurm.dependencies.resolve_dependency` (--depends-on),
`slurm.status.query_node_status` (plan status). They all route through
`_mlflow.latest_run` — one filter shape, one ordering, one place to fix
bugs. `slurm.sizing.estimate_walltime_minutes` is the fourth call site;
it queries 50 historical rows for a group p95, so it stays separate.

## File map

| File | Role |
|---|---|
| `graphids/cli/submit.py` | Atomic CLI flag surface; flag→TLA shaping; `--skip-if-finished`. |
| `graphids/cli/run.py` | Plan CLI (`run` + `status`). |
| `graphids/cli/training.py` | `fit` / `test` Typer commands; `_prepare` for compute-node setup. |
| `graphids/slurm/submit.py` | Library `submit()`; `_TrainingJob` payload + `checkpoint()`. |
| `graphids/slurm/run.py` | `render_plan_script` — toposort + emit bash artifact (no submission). |
| `graphids/slurm/dag.py` | `Node` (Pydantic), `parse_plan`, `toposort`, `filter_with_upstream`. |
| `graphids/slurm/dependencies.py` | `--depends-on` registry + resolution. |
| `graphids/slurm/status.py` | Per-node MLflow status query + table/json formatters. |
| `graphids/slurm/sizing.py` | Optional `--time-from-history` walltime estimation. |
| `graphids/_mlflow.py` | `latest_run`, `is_finished`, `start_training_run`, `log_test_run`, tag/filter helpers. |
| `configs/plans/ofat.jsonnet` | OFAT topology (15 fits + 15 tests + 1 command). |
| `configs/resources/submit_profiles.json` | Raw `submitit.AutoExecutor` kwargs keyed `[mode][cluster][length]`. |
| `scripts/slurm/_preamble.sh` | Module load + venv + `.env` on the compute node. |

## Non-obvious invariants

- **`_TrainingJob.__call__` runs `_prepare()` provider setup** because it
  bypasses Typer's root callback. If `cli.training.fit` is ever
  refactored to require the Typer callback, the compute-node path
  silently regresses.
- **submitit pickles the payload at submission time.** Code edits to
  `graphids/` that land *after* a job is queued won't reach it.
  Cancel + resubmit on bug fixes.
- **`_align_cpus_to_mem` mutates the params dict in place.** Each
  `submit()` call shallow-copies the profile entry first, so the global
  `_PROFILES` is not mutated.
- **MLflow is a hard dep**. Failures propagate from `start_training_run`
  / `log_test_run`. The two soft-failure paths (`log_params` resume
  conflict, `end_training_run` cleanup) are documented in `_mlflow.py`'s
  module docstring.
- **`run_dir` is stamped into the pickled `_TrainingJob`** so SIGUSR2
  recovery doesn't re-render jsonnet on the compute node.
