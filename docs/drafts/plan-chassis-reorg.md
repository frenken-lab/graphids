# Plan-chassis reorganization + plan-id tracking

Status: **SUPERSEDED 2026-05-05.** Steps 1+2 (mechanical reorg + plan-id
schema/CLI) shipped in commits `5d81480` and `9e275de`. A subsequent
readability pass landed `lib.py → primitives.py`, `blueprint.py →
schema.py`, `row.py` folded into `compose.py`, and a `graphids.plan`
public-API surface for plan authors. The framework eval at
`docs/drafts/experiment-framework-evaluation.md` evaluated migration to
Optuna / Ray Tune / Ray Core / Flambe; decision was to keep custom but
adopt seven design constraints documented at
`docs/drafts/chassis-design-lessons.md`. Step 4 (TUI) is killed in
favor of an eventual static-HTML render only if `plans show` outgrows
the CLI.

References to `plan/blueprint.py`, `plan/lib.py`, `plan/row.py` below
are stale (those files no longer exist). Kept for history.

Owners: Robert (rf15)

---

Two distinct lifts, sequenced. Step 1 is mechanical. Step 2 is additive
behavior. Heavy reorg deferred until 2 surfaces real pain.

## Audit of `graphids/plan/` (current state)

```
graphids/plan/                 1110 LOC
├── blueprint.py     (228) — Pydantic schema (Row union, RenderedConfig, BlueprintArray)
├── compose.py       (187) — composer: spec dicts → frozen RowSpec → row dicts
├── lib.py           (157) — class-path registry + spec helpers (mixed concerns)
├── catalog.py       (105) — dataset registry + path math (lake_root, run_dir, …)
├── row.py           (103) — RowSpec: composer-side builder for TrainRow
├── constants.py     ( 38) — PREPROCESSING_VERSION
└── plans/
    ├── ofat.py / unsupervised.py / fusion.py        — TrainRow producers
    └── ops/
        ├── gat_taunorm_smoke.py    — also a TrainRow producer (smoke-flavored)
        └── rebuild_cache.py        — CacheRow producer (data prep, no model)
```

### Problems

1. **`plans/ops/` is incoherent.** Holds both training plans (`gat_taunorm_smoke`)
   and data-prep plans (`rebuild_cache`). The split's actual meaning is
   "things I run rarely" — not a structural distinction.
2. **`config/` is a stale name.** Jsonnet-era baggage (when `configs/`
   held JSON config data). The directory now holds: schema + composer +
   path math + plan modules. `plan/` is more accurate.
3. **`catalog.py` + `constants.py` aren't config.** Both are imported
   from `core/data/datasets/_base.py`, `slurm/submit.py`, `_mlflow.py`,
   `orchestrate.py` — non-config code. Path math + data constants
   should live elsewhere.
4. **`row.py` and `blueprint.py` are tightly coupled** but split.
   `RowSpec` is the composer-side mutable builder; `TrainRow` is the
   Pydantic frozen validator for the dict it emits. Documented split.
5. **`lib.py` mixes registry constants with helper functions.**
   Cosmetic.

### Reorg paths considered

| Tier | Scope | Verdict |
|---|---|---|
| **Light** | resplit `plans/ops/` only | underdoes it — `config/` rename is also worth doing |
| **Medium** | + rename `config/` → `plan/`, move catalog+constants to `paths.py` | **chosen** |
| **Heavy** | + split `lib.py` into registry/specs, merge `row.py` into `blueprint.py` | mostly aesthetic; defer |

Heavy reorg's extra moves (`lib.py` split, `row.py` merge) are
cosmetic — they don't unlock the plan-id tracking feature, and the
existing splits document real distinctions. Defer until something
actually hurts.

## Step 1 — Medium reorg

### Target layout

```
graphids/
├── plan/                       # was config/
│   ├── blueprint.py
│   ├── compose.py
│   ├── lib.py
│   ├── row.py
│   └── plans/
│       ├── ablations/          # was plans/ — concrete experiment sweeps
│       │   ├── ofat.py
│       │   ├── unsupervised.py
│       │   └── fusion.py
│       ├── smoke/              # was plans/ops/ (training-flavored)
│       │   └── gat_taunorm.py  # renamed from gat_taunorm_smoke.py
│       └── data/               # was plans/ops/ (data-prep)
│           └── rebuild_cache.py
├── paths.py                    # was config/{catalog,constants}.py — merged
└── …
```

### File moves

| From | To |
|---|---|
| `graphids/plan/` | `graphids/plan/` |
| `graphids/plan/catalog.py` | `graphids/paths.py` (merged with constants) |
| `graphids/plan/constants.py` | `graphids/paths.py` (merged) |
| `graphids/plan/plans/ofat.py` | `graphids/plan/plans/ablations/ofat.py` |
| `graphids/plan/plans/unsupervised.py` | `graphids/plan/plans/ablations/unsupervised.py` |
| `graphids/plan/plans/fusion.py` | `graphids/plan/plans/ablations/fusion.py` |
| `graphids/plan/plans/ops/gat_taunorm_smoke.py` | `graphids/plan/plans/smoke/gat_taunorm.py` |
| `graphids/plan/plans/ops/rebuild_cache.py` | `graphids/plan/plans/data/rebuild_cache.py` |

### Import sweep

Global sed (order matters, most-specific first):

```bash
find graphids tests scripts -name '*.py' -not -path '*/__pycache__/*' -print0 \
  | xargs -0 sed -i \
    -e 's/graphids\.config\.catalog/graphids.paths/g' \
    -e 's/graphids\.config\.constants/graphids.paths/g' \
    -e 's/graphids\.config\.plans\.ops\.gat_taunorm_smoke/graphids.plan.plans.smoke.gat_taunorm/g' \
    -e 's/graphids\.config\.plans\.ops\.rebuild_cache/graphids.plan.plans.data.rebuild_cache/g' \
    -e 's/graphids\.config\.plans\.ops/graphids.plan.plans.smoke/g' \
    -e 's/graphids\.config\.plans\.\(ofat\|unsupervised\|fusion\)/graphids.plan.plans.ablations.\1/g' \
    -e 's/graphids\.config\./graphids.plan./g'
```

### Plan invocation surface change

`graphids run` plan-name argument changes from
`smoke.gat_taunorm` → `smoke.gat_taunorm`,
`data.rebuild_cache` → `data.rebuild_cache`,
`ofat` → `ablations.ofat`. Update `cli/commands.py` import string and
docstrings, and `scripts/cache_rebuild.sh`, `cli/test.py`.

### Verification

```bash
python -c "from graphids.plan.blueprint import BlueprintArray, CacheRow; \
           from graphids.plan.compose import compose; \
           from graphids.paths import lake_root, run_dir, PREPROCESSING_VERSION; \
           import graphids.plan.plans.ablations.ofat; \
           import graphids.plan.plans.smoke.gat_taunorm; \
           import graphids.plan.plans.data.rebuild_cache; \
           print('OK')"
python -m graphids run smoke.gat_taunorm --dataset hcrl_sa --seed 42 -o /tmp/v.json
python -m graphids run data.rebuild_cache --dataset hcrl_sa --seed 42 -o /tmp/v2.json
```

### Rules updates

- `.claude/rules/single-submission-primitive.md` — references to
  `graphids/plan/plans/ops/` → `graphids/plan/plans/`.
- `.claude/rules/config-system.md` — file paths.
- `.claude/rules/data-layout.md` — `graphids/plan/catalog.py` →
  `graphids/paths.py`.
- `CLAUDE.md` — same ~6 references.
- `docs/reference/config-architecture.md` — full path sweep.

Net diff: ~30 file moves, ~80 import-line edits, ~10 doc/rule edits.
No behavior change.

## Step 2 — Plan-id grouping

A `plan_id` woven through every render → submit → log → query so a
single render is a queryable group.

### Concept

```
graphids run smoke.gat_taunorm …  →  mints plan_id (uuid7)
                                  →  writes {plan_id, plan_module, dataset, seed,
                                              created_at, rows: [...]} to JSON
                                  →  every row carries plan_id

graphids submit  →  threads plan_id into:
                   - sbatch --comment "graphids.plan_id=<id>"
                   - MLflow tag graphids.plan_id=<id>

graphids plans list           →  distinct plan_ids from MLflow tags (last 7d)
graphids plans status <id>    →  read-only summary across all rows in plan
```

### Schema changes (`graphids/plan/blueprint.py`)

```python
class Plan(_StrictModel):
    """Top-level rendered plan — was a bare list[Row]."""
    plan_id: str          # uuid7 for time-sortability (sortable lex == sortable temporal)
    plan_module: str      # e.g. "smoke.gat_taunorm"
    dataset: str
    seed: int
    created_at: str       # ISO8601, UTC
    rows: list[Row]

# Each row gains:
class _RowMixin:
    plan_id: str          # shared across all rows in one render
```

`BlueprintArray` is replaced by `Plan` (kept for back-compat as a
RootModel alias for one cycle, then removed).

### Render (`graphids/cli/commands.py::run_cli`)

```python
plan_id = uuid7().hex  # via uuid_extensions or a 12-line uuid7 impl
rows = mod.build(dataset=dataset, seed=seed)
for r in rows:
    r["plan_id"] = plan_id
plan = Plan.model_validate({
    "plan_id": plan_id,
    "plan_module": plan,
    "dataset": dataset,
    "seed": seed,
    "created_at": datetime.now(UTC).isoformat(),
    "rows": rows,
})
```

`graphids run` now writes a `Plan` object (not a bare array). `jq -c '.rows[]'`
replaces `jq -c '.[]'` in scripts. Update `scripts/cache_rebuild.sh` and
`cli/test.py::smoke`.

### Submit (`graphids/slurm/submit.py::submit_row`)

```python
sbatch_kwargs["comment"] = f"graphids.plan_id={row.plan_id}"
```

`SlurmProvider` already accepts a `comment` kwarg → `--comment` sbatch
directive. `sacct -o Comment` exposes it.

### MLflow tag (`graphids/_mlflow.py::start_training_run`)

```python
mlflow.set_tag("graphids.plan_id", row.plan_id)
```

`identity_tags()` already builds the per-row tag dict — add
`graphids.plan_id` there so it's set once per phase.

### Tracking commands (`graphids/cli/plans.py` — new)

```python
@plans_app.command("list")
def list_plans(days: int = 7) -> None:
    """Distinct plan_ids active in the last N days, newest first."""
    # mlflow.search_runs across all graphids/* experiments,
    # filter on graphids.plan_id existence, group by plan_id, sort by max(start_time).

@plans_app.command("status")
def plan_status(plan_id: str) -> None:
    """Per-row status summary for one plan_id."""
    # mlflow.search_runs(filter_string=f"tags.graphids.plan_id = '{plan_id}'")
    # sacct --comment="graphids.plan_id={plan_id}" --format=JobID,State
    # zip on row.name, render as table.
```

Both are pure read queries — `single-submission-primitive.md` allows
this (rule blocks read-and-*act* helpers, not read-only views).

### Why uuid7

- Time-sortable lexicographically (debugging: ls + tail = recent plans first).
- 128-bit, no collision risk across users/clusters.
- Already implementable in 12 lines without a new dep, or via
  `uuid-utils` (single-purpose dep, ~10KB).

### Verification

1. Render a plan → assert every row's `plan_id` is identical and the
   top-level `plan_id` matches.
2. Submit one row → `sacct -j <jid> -o Comment` shows the plan_id.
3. Run a fit → MLflow run has `tags.graphids.plan_id = <id>`.
4. `graphids plans status <id>` returns row-level table within 2s.

## Step 3 — Reassess heavy bits

After step 2 lands, decide:

- **`lib.py` split** — does mixing class-path strings with helper
  functions block plan-id work? (Probably not.)
- **`row.py` ↔ `blueprint.py` merge** — does the split make adding
  `plan_id` to both `RowSpec` and `TrainRow` painful? (Maybe — both
  need the field.)

Only act on these if step 2 surfaces concrete pain. Otherwise leave them.

## Out of scope

- Per-row resume state in `Plan` JSON. SLURM handles that
  (preempt-resume via SIGUSR2 + `auto_requeue=True`).
- Any "pipeline driver" that walks plan rows and submits — explicitly
  blocked by `single-submission-primitive.md`. The user/LLM still
  iterates `jq | submit`. Plan-id only adds *grouping*, not *driving*.
- Backwards compatibility for old plan JSONs without `plan_id`. We
  migrate or re-render; the JSON files are ephemeral.

## Risks

1. **Import sweep misses an edge case.** Mitigation: `python -c
   "import graphids; …"` smoke after sed; `grep -r 'graphids\.config'`
   to confirm zero residuals.
2. **MLflow tag cardinality.** Per-plan tags are unbounded over time.
   Acceptable for now (existing per-run tags already are).
3. **`Plan` schema break.** Existing `plan.json` files in scratch dirs
   become unreadable. Acceptable — they're transient render artifacts.

## Effort estimate

- Step 1: ~1hr (mechanical moves, mechanical sed, doc sweep)
- Step 2: ~2hrs (schema, render, submit threading, two new commands,
  tests)
- Step 3: dependent, only if needed
