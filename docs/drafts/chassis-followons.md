# Plan-chassis follow-ons

Status: DRAFT — 2026-05-05 (rewritten after framework eval surfaced
that the original "do now" was chassis growth not justified by the
actual reproduction contract).

Owners: Robert (rf15)

Original audit surfaced ~5 candidate lifts. Re-evaluated against the
framework eval (`experiment-framework-evaluation.md`) and design
lessons (`chassis-design-lessons.md`). The reformed shortlist deletes
or shrinks most of them.

---

## What changed in the analysis

The original "do now" proposed:

1. Durable `${RUN_ROOT}/plans/<plan_id>/plan.json`
2. `${RUN_ROOT}/plans/<plan_id>/jobs.jsonl` with row lifecycle state
3. `graphids plans retry <plan_id> <row_name>`
4. `graphids plans show <plan_id>`

The framework eval surfaced three findings that overturn most of this:

- **Reproduction contract is `git SHA + plan module + plan args`**, not
  the rendered JSON. The JSON is intermediate cache. `git checkout
  <sha> && graphids run <plan> --dataset X --seed Y` regenerates the
  plan deterministically.
- **MLflow IS the trial-state store.** `tags.graphids.plan_id` +
  `attributes.status` is queryable. SLURM has the rest via sacct.
  Adding `jobs.jsonl` would be a *third* parallel state store.
- **Lesson 8: be cautious growing the chassis.** The chassis is what
  we'd delete on Optuna migration. The original "do now" was pure
  chassis growth.

Applied honestly: the durable plan store and `plans retry` either
duplicate state MLflow already has, or replicate functionality that
falls out of the existing render-then-submit composition.

---

## Reformed "do now" — ~50 LOC, no new state stores

### 1. `graphids run --filter <name-glob>` (~10 LOC)

Render only the rows whose `name` matches the glob. Single-row render
becomes the unit of retry — no new CLI command needed:

```bash
# Retry a specific failed row, keyed by name
graphids run ablations.ofat --dataset hcrl_sa --seed 42 --filter 'gat_focal*' \
    | jq -c '.rows[]' \
    | xargs -I{} graphids submit --row '{}' --cluster pitzer
```

The user/LLM iterates (per `single-submission-primitive.md`). The
filter mechanic just shrinks the rendered set. Replaces the
`plans retry` use case via composition.

**Why this is better than `plans retry`:**

- No durable plan.json required (the reproduction contract is git+module+args).
- No "store doesn't exist" warning paths.
- One CLI surface (`run`) gains one optional flag, instead of a new
  `plans retry` command + parallel code path.
- Survives chassis migration cleanly — Optuna's analogue is just
  filtering trials at submit time.

### 2. `graphids plans show <plan_id>` (~30 LOC) — read-only MLflow query

Thin wrapper around `MlflowClient.search_runs(filter_string="tags.\`graphids.plan_id\` = '<id>'")`.
Renders a Rich table:

```
row_name              | run_id    | status      | started        | best_metric
gat_focal_seed42_fit  | abc123    | FINISHED    | 2026-05-05...  | val_auroc=0.847
gat_focal_seed42_test | def456    | RUNNING     | 2026-05-05...  | —
gat_ce_seed42_fit     | ghi789    | FAILED      | 2026-05-05...  | val_auroc=0.612
```

**Read-only.** No state file, no append paths, no filelock complications.
MLflow's run state is the single source of truth.

If MLflow doesn't have a tag we want shown (e.g., the SLURM jid),
add it as a tag on the run from `_mlflow.start_training_run` —
don't introduce a parallel store.

### 3. Verify reproduction-contract MLflow tags exist (~10 LOC if needed)

The reproduction contract is `git SHA + plan module + plan args`. Verify:

- `tags.graphids.plan_id` ✓ (already wired commit `9e275de`)
- `tags.graphids.git_sha` — verify `_mlflow.start_training_run` sets it; add if not
- `tags.graphids.plan_module` — set by `graphids exec` (the row carries
  `plan_id`, but does it carry the module name?). Verify; add if not.
- `tags.graphids.plan_args` — `dataset`, `seed`, any other `build()`
  args. Verify; add if not.

These tags make `plans show` informative AND make
`git checkout <sha> && graphids run <module> --dataset X --seed Y`
discoverable from the MLflow UI.

### Total

~50 LOC, no new directory under `${RUN_ROOT}`, no parallel state file,
no Pydantic schema changes, single-submission-primitive untouched.

---

## What was killed (and why)

### ~~Durable `${RUN_ROOT}/plans/<plan_id>/plan.json`~~

**Original justification:** "today plan_id lives only as MLflow tags +
sacct comments. Both expire. The rendered plan JSON is `mktemp` scratch."

**Reformed analysis:**

- Drift resistance (the row JSON in the sbatch) doesn't depend on a
  stored plan.json — the row in the sbatch script is the load-bearing
  artifact, and that's already there.
- "60-day-old plan, MLflow tags survive, sacct gone, plan.json gone"
  → if MLflow tags survive (and they do — `mlflow.db` is on shared
  NFS, not /tmp), and they include `git_sha + plan_module +
  plan_args`, you can regenerate the plan exactly. The plan.json
  adds no new information.
- The only case where stored plan.json wins is "MLflow DB lost AND
  git history corrupted." Defensive against an unrealistic failure mode.

**Killed.**

### ~~`${RUN_ROOT}/plans/<plan_id>/jobs.jsonl` with lifecycle state~~

**Original justification:** "Lesson 2 (borrow Optuna's Trial vocabulary
— state enum)."

**Reformed analysis:** Lesson 2 said *borrow the vocabulary*, not
*build a parallel state store*. Optuna's storage IS the state. For
us, MLflow IS the state. Adding `jobs.jsonl` introduces a third
state location (alongside MLflow and sacct) that has to be kept
in sync. Pure chassis growth.

**Killed.** State queries go through `_mlflow.build_search_filter(...)`
+ sacct comment scan as today.

### ~~`graphids plans retry <plan_id> <row_name>`~~

**Original justification:** "preserves single-submission-primitive
because it submits one row."

**Reformed analysis:** the rule blocks "read-and-act helpers" that
reduce N submits to one CLI command via stored state. `plans retry`
reads a stored plan.json, finds a row, submits it. That's exactly
the shape the rule blocks — it's just hiding behind "but only one
row at a time." Replace with `--filter` + iteration; the user/LLM
loop is the iteration mechanism, as the rule intends.

**Killed.** `--filter` on `run` is the replacement.

### ~~Row dependencies in plan output (`depends_on_row_name`)~~ — KILLED 2026-05-05

> Originally proposed: an advisory `depends_on_row_name: list[str]`
> field on each row, with `plans show` warning about un-submitted
> upstream deps. Sold as "advisory, not a runner."

Killed by the framework eval (Lesson 6 in
`docs/drafts/chassis-design-lessons.md`). Once row-graph metadata
exists in the plan, *something* will eventually iterate and submit
in topo order — that's a DAG runner, which violates
single-submission-primitive (now also documented under "Drift
resistance" in `.claude/rules/single-submission-primitive.md`).
"Advisory" is the on-ramp.

**What stays.** Dependencies between rows continue to be expressed at
submit time via `--depends-on-afterok <jid>` (data dep) /
`--depends-on-afterany <jid>` (preempt-resume chain), captured from
the prior submit's stdout. Each row's existing `upstreams` field
already declares its data dependencies on ckpt paths — that's enough
metadata for a future visualization without inviting the runner.

---

## Do later

### Named plans (`--name`)

`graphids run` mints uuid7. Re-running the same plan after Ctrl-C
splits state across two plan_ids. Add `--name` flag:

```python
@app.command("run")
def run_cli(plan, dataset, seed, name: str | None = None, output=None):
    plan_id = name or mint_plan_id()
```

**Cost:** ~3 LOC. **Why later:** today's failure mode is "I lost
the plan.json" — already addressed by the reproduction contract
(git+module+args). `--name` is polish for grouping multiple
re-renders under one ID. Defer until shown to be needed.

### MLflow median-pruning callback (Lesson 4)

`graphids/core/callbacks/MLflowMedianPruningCallback` (~40 LOC).
Worker-local query against MLflow; if this run's `val_auroc` at
epoch N is below the median of completed peers in the same plan_id,
raise `KeyboardInterrupt` for clean shutdown. Gated by env var.

**Why later:** real GPU-hour win, but only worth it on sweeps wasting
>20% of compute on clearly-bad trials. Implement when that pressure
shows up, not preemptively (per Lesson 4: don't refactor before pain).

### Static-HTML dashboard (Lesson 5)

If `plans show` CLI outgrows itself, target a jinja+plotly static
render at `${RUN_ROOT}/plans/<plan_id>/dashboard.html`. No service.
Drop the TUI direction entirely.

**Why later:** the CLI table is enough until proven otherwise.

---

## Explicitly **not** doing

- **`graphids launch X --cluster Y`** (render+submit in one): violates
  atomicity guarantee on `submit_row`; bash composition is the right
  level. The `--filter` flag is the closest acceptable adjacent.
- **DAG runner for row dependencies**: explicit single-submission-
  primitive violation. Even an "advisory" version is the slope toward
  one — see the killed subsection above.
- **TUI for plan management**: Lesson 5. Static-HTML if anything.
- **Drop `plan_args`**: was motivated by the TUI re-submission flow.
  TUI dropped → reconsider. Probably also droppable since
  `tags.graphids.plan_args` on MLflow runs covers it; leave the
  `Plan.plan_args` field for now and revisit on a future cleanup.
- **Replace MLflow with Aim/ClearML**: locked in by infra.
- **Hand-rolled JSON-IPC layer**: Pydantic at the JSON boundary pays for
  itself.

---

## Sequencing recommendation

1. **`graphids run --filter <glob>`** (~10 LOC) — unblocks
   single-row retry via composition.
2. **MLflow tag verification** (`git_sha`, `plan_module`, `plan_args`)
   — verify and patch (~10 LOC if any are missing).
3. **`graphids plans show <plan_id>`** (~30 LOC) — read-only MLflow
   query wrapper.
4. **Use the chassis on a real OFAT sweep.** Capture pain points.
5. **Named plans (`--name`)** only if step 4 surfaces it.
6. **Pruning callback** (Lesson 4) when GPU-hour pressure justifies.
7. **Static-HTML dashboard** only if step 3 outgrows itself (Lesson 5).

---

## Risks

1. **`--filter <glob>` makes "submitted partial sweep" the default
   failure mode.** Mitigation: `plans show` reports which rows in the
   plan_module's full render have no MLflow run yet. User catches the
   gap by inspection.
2. **MLflow tags drift** (e.g., a future schema change renames
   `graphids.plan_args`): the reproduction contract degrades silently.
   Mitigation: keep the tag-name list in `_mlflow.IDENTITY_KEYS` (one
   place to update), and `plans show` warns when a tag is missing on
   a row.

---

## What this rewrite is

A correction. The original "do now" rationalized chassis growth that
the framework eval explicitly warned against. The reformed version
deletes ~80 LOC of proposed code and pushes the remaining ~50 LOC
through MLflow + composition rather than a new state store. The
single-submission-primitive rule is preserved literally.
