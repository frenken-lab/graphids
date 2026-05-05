# Chassis design lessons from the framework evaluation

**Status:** 2026-05-05. Source: `experiment-framework-evaluation.md`.
**Decision:** keep custom chassis. This doc captures what the eval taught
us about HOW to evolve it.

---

## Core vs chassis — the dividing line to defend

The eval's "what survives any choice" section split the codebase in two:

**Core** (framework-agnostic, keep growing):

- `plan/compose.py` — block assembly
- `plan/primitives.py` — class-path catalog + leaf builders
- `paths.py` — `run_dir` / `best_ckpt` / `states_dir`
- `orchestrate.run_row` — importlib + Lightning dispatch
- MLflow callback (`_mlflow.py`)
- SIGUSR2 preempt-resume in `orchestrate._trainer_kwargs`

**Chassis** (the bespoke study/trial layer):

- `plan/schema.py` — `Plan` / `TrainRow` / etc.
- `plan_id` minting + sacct/MLflow tag wiring
- `cli/plans.py`
- Drafted `plans show` / `plans retry` followons
- `plan_args` field

Every future change should ask: **am I extending the core, or growing the
chassis?** The chassis is what we'd delete on Optuna migration. Adding
to it is opting into more code we own forever.

---

## Lesson 1 — name the load-bearing property explicitly

The eval surfaced that **drift resistance** is what justifies the JSON
artifact, not the diff/stash/hand-over use cases (those evaporate on
inspection). Today drift resistance exists as a side effect of "row
JSON is in the sbatch script."

The asymmetry is by design:

|  | code change reaches queued jobs? | config change reaches queued jobs? |
|---|---|---|
| Custom | ✓ (sbatch re-imports source) | ✗ (row JSON frozen) |

**Action:** add a paragraph to `.claude/rules/single-submission-primitive.md`:
> The rendered row JSON is frozen at `graphids run` time. Edits to
> `compose()` or trainer defaults after submit do NOT reach queued
> jobs — this protects in-flight sweeps from accidental drift. Edits
> to model/data class internals DO reach queued jobs (the sbatch
> re-imports source at exec time). The asymmetry is intentional.

Without this written down, a future refactor might break the property
incidentally and nobody will notice until a paper is wrong.

---

## Lesson 2 — borrow Optuna's Trial vocabulary

Even keeping our schema, Optuna's Trial data model is a more
thought-through factoring than ours. **Important:** the lesson is
*borrow the vocabulary*, NOT *build a parallel state store*. Optuna's
storage IS the state. For us, **MLflow IS the state.** Don't add a
third store alongside MLflow + sacct.

| Optuna Trial | Custom TrainRow today | What we should learn |
|---|---|---|
| `params` (searched dims) | rolled into `rendered_config` | distinguish "axes the plan varies" from "everything else" — informs the search-space refactor in Lesson 3 |
| `user_attrs` (free-form) | spread across `meta` + `identity` + `resources` | one typed bucket + one free-form bucket beats three overlapping ones |
| `state` enum (RUNNING/COMPLETE/FAILED/PRUNED) | live in MLflow run state + sacct comments | **don't replicate it.** Read from MLflow when you need state. Add tags to MLflow runs if a state we care about isn't surfaced (e.g., row name, plan module) |
| `intermediate_values` | logged via `mlflow.log_metric(..., step=epoch)` | already mostly there; needed for pruning (Lesson 4) |
| `trial.number` (per-study ordinal) | row `name` only | useful concept; cheap to add as an MLflow tag on the run, NOT as a new artifact on disk |

**Action:** verify the reproduction-contract tags exist on every fit
run (`graphids.plan_id`, `graphids.plan_module`, `graphids.plan_args`,
`graphids.git_sha`, `graphids.row_name`). Add any missing.
`plans show` becomes a thin `MlflowClient.search_runs` query against
those tags. **No `${RUN_ROOT}/plans/<plan_id>/jobs.jsonl` file.**
The original "do now" had this; it's killed in the reformed
`chassis-followons.md`.

---

## Lesson 3 — separate "what to search" from "how to render one trial"

Today `build()` does both: enumerates the Cartesian product AND renders
each combo. Fine for grid; doesn't scale to TPE/Bayesian/random and
makes the search space un-inspectable without running the whole render.

Optuna's `objective(trial)` factors these:

```python
def search_space() -> dict[str, list]:           # declarative
    return {"loss_fn": ["focal", "ce", "weighted_ce"], "seed": [42, 43, 44]}

def render_one(params, *, dataset) -> dict:      # one trial
    loss = LOSSES[params["loss_fn"]]
    return compose(
        model=spec(GAT, loss=spec(loss)), data=..., meta={...}
    ).fit(...)

def build(*, dataset, seed):                     # iteration is generic
    return [render_one(combo, dataset=dataset)
            for combo in itertools.product_dict(search_space())]
```

Pure refactor — same JSON output, drift-resistant, search space becomes
declaratively inspectable (`graphids plans search-space <plan>` is trivial).
Also sets us up to swap in non-grid samplers later without rewriting plans.

**Action:** convert one plan as a spike (`ablations/ofat.py` — its
`SWEEPS` dict is already 80% of the way there). If it cleans up the
plan, propagate. If it complicates more than it clarifies, leave the
others as-is.

---

## Lesson 4 — pruning fits single-submission-primitive cleanly

Optuna's pruners are worker-local: the worker queries shared storage for
peer state and self-terminates. **Architecturally, we can do the same
with MLflow as the storage.** No central scheduler, no rule violation.

**Action:** add `MLflowMedianPruningCallback` (~40 LOC) under
`graphids/core/callbacks/`:

- `on_validation_epoch_end`: query MLflow for `val_auroc` at this epoch
  across COMPLETED runs in the same `plan_id` / `group`.
- if this run's metric < median of peers at this epoch, raise
  `KeyboardInterrupt` (Lightning catches → clean shutdown, MLflow run
  marks as KILLED).
- gated by env var `GRAPHIDS_PRUNE=1` so existing runs don't break.

Real GPU-hour win on long sweeps. Same pattern Optuna uses; lifted
directly without adopting the framework.

---

## Lesson 5 — the dashboard impulse is right; the form was wrong

The drafted `plans show` is a CLI two-table render; the proposed TUI
(step 4 in `plan-chassis-reorg.md`, never drafted) pushed further into
terminal land. The eval showed the value isn't terminal-bound — it's
parallel-coordinates, hyperparameter importance, filterable trial views.
Optuna-dashboard works because it ships those.

**Action:**

- Keep `plans show` as the quick CLI summary. Don't grow it into a TUI.
- If observability graduates, target a **static HTML render** at
  `${RUN_ROOT}/plans/<plan_id>/dashboard.html`: jinja template + plotly
  for parallel-coordinates + a small run table. No service to run;
  open with `xdg-open` or `python -m http.server`. Fits the project's
  "no service infra" posture.
- Drop the TUI direction entirely (the proposed file was never drafted).

---

## Lesson 6 — kill the `depends_on_row_name` direction

The chassis-followons "do later" item adding advisory
`depends_on_row_name` on rows is the slope toward a DAG runner. Once
deps are in the data, *something* will eventually iterate and submit
in topo order. That's a pipeline driver, which violates
single-submission-primitive.

**Action:** delete the `depends_on_row_name` direction from
`chassis-followons.md`. Dependencies stay as today: the user passes
`--depends-on-afterok <jid>` to `graphids submit`, captured from the
prior submit's stdout. If row-graph visibility is ever wanted, render
it from `build()`'s structure — each row's `upstreams` field already
declares its data dependencies on ckpt paths.

---

## Lesson 7 — `build()` is the primary artifact, not the JSON

Reproduction is `git SHA + plan module + args`, not the rendered JSON
file. The JSON is intermediate cache. Future plan code should be
**prioritized for `build()` readability** — it's what people actually
read to understand what was run.

Style for `graphids/plan/plans/*.py`:

- The variant table / search space visible at the top of `build()`.
- Variant names are the dict keys, not generated strings.
- One plan per ablation axis when feasible; subsumption (OFAT absorbing
  the prior `supervised` and `supervised_ablations` plans) is good.
- Comments explain *why* an axis exists, not *what* it does.

`ablations/ofat.py` already does this well. Codify it as a contributing
note rather than letting it drift on subsequent plans.

---

## Lesson 8 — distinguish CLI-shape rules from architectural invariants

The original `single-submission-primitive.md` rule was about CLI shape
("N jobs = N invocations of `graphids submit`"). The framework eval
treated it as architectural; **it was actually a discipline rule
protecting four real architectural properties**: render purity, drift
resistance, MLflow-as-state, reproduction-contract tags. Once those
four were named explicitly (`chassis-invariants.md`, 2026-05-05), the
CLI-shape rule could be killed and `graphids plans submit` (multi-row,
MLflow-aware) shipped without breaking anything.

**Action:** when proposing or rejecting a CLI verb, ask:
1. Does it break render purity? (Render+submit fusion = yes.)
2. Does it break drift resistance? (Re-running `compose()` at exec = yes.)
3. Does it write a parallel state store? (jobs.jsonl = yes.)
4. Does it skip the reproduction-contract tags? (Delete a tag = yes.)
5. Does each row's outcome have its own log line? (Hidden batch loop = no.)

If 1–4 are no and 5 is yes → it's allowed. The CLI surface is open;
the architectural surface is closed.

This is the failure mode previously named in
`feedback_evaluate_libraries_on_own_terms.md`: when a subsystem
*resembles* a known shape, ask which of its rules are architectural
and which are convention. Conflating them blocks otherwise-fine
ergonomic features in the name of an architecture they don't actually
threaten.

---

## Hold-list update

Re-evaluate each draft doc against these lessons before resuming:

| Draft | Status post-lessons |
|---|---|
| `chassis-followons.md` original "do now" (durable plan store + `jobs.jsonl` + `plans retry`) | **Killed.** Reformed to ~50 LOC: `run --filter <glob>` + read-only `plans show` via MLflow + verify reproduction-contract tags. No new state stores. |
| `chassis-followons.md` reformed "do now" (filter + plans show + tag verification) | **Ship next.** ~50 LOC, no chassis growth. |
| `chassis-followons.md` "do later" `--name` | Keep, low priority |
| `chassis-followons.md` "do later" `depends_on_row_name` | **Killed** (Lesson 6) |
| `chassis-followons.md` "drop `plan_args`" | Drop — not needed; TUI dropped; tag covers it |
| `plan-chassis-reorg.md` | Superseded by what shipped 2026-05-05 |
| TUI direction (proposed in `plan-chassis-reorg.md` step 4, never drafted) | **Killed** (Lesson 5). Static-HTML render only if `plans show` outgrows CLI. |

---

## What this is not

- Not a roadmap. These are design constraints to apply when work in this
  area resumes; some may never happen.
- Not a license to refactor proactively. Lessons 3 and 4 are spikes; do
  them when the pain shows up, not preemptively.
- Not exhaustive. The framework eval surfaced these; future bugs will
  surface more.
