# Chassis Invariants

The CLI shape may grow (multi-row verbs, resume flags, batch operations).
**These four properties stay.** Each protects something concrete; breaking
one breaks reproduction or correctness.

> Replaced `single-submission-primitive.md` on 2026-05-05. The old "N jobs
> = N invocations" rule was killed once `graphids plans submit` (multi-row
> with `--resume`/`--filter`/`--dry-run`) was deemed worth shipping. The
> architectural properties below are what that rule was really protecting;
> they survive without the CLI-shape constraint.

## 1. Render is pure JSON; submit consumes it

`graphids run` outputs validated JSON. It does NOT submit, query MLflow,
or touch SLURM. Submission is a separate step (or verb).

**Why:** typos / missing kwargs / schema drift surface at render time on
the login node, before any sbatch. Fusing render+submit into one verb
(e.g., `graphids launch <plan>`) hides the validation gate; a typo would
blow up after N partial submits.

**What this blocks:**
- `graphids launch <plan> --cluster X` (render+submit fused).
- Render emitting an executable bash script. JSON only — data, not action.
- `graphids run` reaching out to MLflow / SLURM / network.

## 2. Drift resistance — row JSON frozen at submit time

The sbatch body is the literal `python -m graphids exec --row '<json>'`.

| Edit between submit and exec | Reaches queued jobs? |
|---|---|
| Model / data class internals (e.g. `GAT.forward`) | ✓ — sbatch re-imports source |
| `compose()` / primitives / trainer defaults | ✗ — row JSON frozen in sbatch |

**Why:** a queued sweep can run for days. A stray edit to a `compose()`
default must NOT silently change a queued config — or paper results
become "what was actually run on which seed?" The asymmetry is intentional:
code-internal fixes (a forward-pass bug) DO reach queued jobs; config
defaults DON'T.

**What this blocks:**
- A "resubmit-from-plan-id" feature that re-renders at exec time.
- Reading the plan back into Python at submit time and re-running
  `compose()` for "the latest version."
- Sbatch bodies that call `graphids run` instead of `graphids exec`.

## 3. MLflow is the trial-state store

Trial state — RUNNING / FINISHED / FAILED, params, metrics, lineage —
lives in `mlflow.db`. **No parallel `jobs.jsonl`, no
`${RUN_ROOT}/plans/<plan_id>/` state directory.** Multi-row verbs read
MLflow at decision time; they do not maintain their own state.

**Why:** every parallel state store eventually drifts. The 2026-04
`metrics.jsonl` deletion was about exactly this. MLflow already has
the schema (Trial state, intermediate values, run lineage) and the
query path (`MlflowClient.search_runs` + `build_search_filter`). Adding
a second store doubles the bug surface for no information gain.

**What this blocks:**
- Stateful runners that write to `${RUN_ROOT}/plans/<id>/jobs.jsonl`.
- A `plans cancel` that maintains a separate "intended-cancelled" list.
- Caching MLflow results between `plans show` invocations.

**What this allows:**
- `plans submit --resume` reading MLflow for finished rows. Read-only on
  state, write-only via `submit_row` (each row's MLflow run is opened
  by Lightning when fit starts).

## 4. Reproduction contract — five MLflow tags

Every fit/test run carries `graphids.{plan_id, plan_module, plan_args,
git_sha, row_name}`. Together they encode:

```bash
git checkout <git_sha>
graphids run <plan_module> --dataset <plan_args.dataset> --seed <plan_args.seed> \
    --filter <row_name> -o - \
  | jq -c '.rows[]' \
  | xargs -I{} graphids submit --row '{}' --cluster pitzer
```

This regenerates the row exactly. Lose a tag → silent reproduction break.

**Why:** the JSON file is intermediate cache, not the contract. The
contract is `git_sha + plan_module + plan_args + row_name`. These five
tags are the on-MLflow representation of that contract.

**What this blocks:**
- Removing any of those tags from `_mlflow.identity_tags(...)`.
- Renaming tag keys without updating `_TAG_KEYS` and migration tooling.
- Per-axis experiments that don't include `plan_id` for cross-cutting
  queries.

---

## What this does NOT block (post-2026-05-05)

CLI shape is now negotiable. The following are **allowed** as long as
each row's submission is independently transparent (one log line per
submit) and independently failable:

- **Multi-row CLI verbs** (`graphids plans submit`, future `plans cancel`,
  `plans wait`, etc.).
- **Batch / resume semantics** with filters
  (`--filter <glob>`, `--resume`, `--skip-finished`, `--include-failed`).
- **Read-and-act helpers** that read MLflow (state store) and submit.
- **Same-batch dependency chains** via `--depends-on-afterok` /
  `--depends-on-afterany` on individual `submit` calls.

## What this still blocks

- **DAG runners with implicit dependency awareness.** Auto-chaining
  `--depends-on-afterok` from each row's `upstreams` field is its own
  large design (failure semantics, partial completion, replay
  granularity); it does not get smuggled in as a `plans submit` flag.
  Implementation note: scanning `upstreams` to *validate* a submission
  order is fine; *automatically chaining* is not.
- **Render+submit fusion** — render must be inspectable before submit.
- **Stateful runners that write parallel state stores.** State queries
  read MLflow; state mutations are scoped to the MLflow run that the
  Lightning logger opens.
- **Pickle-based job invocation.** Sbatch bodies are literal commands;
  no closure pickle, no stale-pickle bug.

## Decision rule

If you're tempted to add something that:
- re-runs `compose()` at submit time / exec time (violates 2), OR
- writes a parallel state store under `${RUN_ROOT}/plans/<id>/` (violates 3), OR
- removes / renames an `identity_tags` field without migration (violates 4), OR
- emits a bash script that orchestrates jobs in bulk (violates 1), OR
- auto-chains `afterok` from the plan's row graph (DAG-runner)

→ **Stop.** None of those are gated by the now-killed CLI-shape rule;
they break the architectural invariants directly.

If you're tempted to add a multi-row verb (`plans X`):
- ✓ if it reads MLflow as the source of truth.
- ✓ if each row's outcome is its own log line.
- ✓ if `--dry-run` prints the would-be actions without executing.
- ✗ if it auto-chains dependencies or writes a parallel ledger.
