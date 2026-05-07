# Plan-chassis follow-ons

Status: DRAFT — 2026-05-05 (rewritten after framework eval). See parent
`chassis-design-lessons.md` for the rationale; this file owns the
actionable residue.

Owners: Robert (rf15)

---

## Shipped (do-now items)

- `graphids run --filter <glob>` — single-row retry via composition. Done.
- `graphids plans show <plan_id>` — read-only `MlflowClient.search_runs`
  wrapper. Done in `90ec01a`.
- Reproduction-contract MLflow tags (`plan_id`, `plan_module`, `plan_args`,
  `git_sha`, `row_name`) — wired in `9e275de` / `90ec01a`; now codified
  as Invariant 4 in `.claude/rules/chassis-invariants.md`.

The original "do now" (durable `plans/<plan_id>/plan.json`,
`jobs.jsonl`, `plans retry`, `depends_on_row_name`) was killed: each
duplicated state MLflow already holds, or seeded a DAG runner. See
parent doc for the kill rationale.

---

## Hold list (do later)

Frozen pending an Optuna-spike conversation that hasn't been scheduled.
Don't pick these up without revisiting the spike question first — the
chassis is what we'd delete on Optuna migration, so net-new chassis
LOC needs justification beyond "would be nice."

### Named plans (`--name`)

`graphids run` mints uuid7. Re-running after Ctrl-C splits state across
two `plan_id`s. ~3 LOC patch on `run_cli`.

**Revisit when:** the `lost-the-plan_id` failure mode shows up in
practice. Today the reproduction contract (git+module+args) covers
re-derivation; `--name` is grouping polish.

### MLflow median-pruning callback (parent Lesson 4)

`MLflowMedianPruningCallback` (~40 LOC) under `graphids/core/callbacks/`.
Worker-local query against MLflow; if `val_auroc` at epoch N is below
the median of completed peers in the same `plan_id`, raise
`KeyboardInterrupt`. Gated by `GRAPHIDS_PRUNE=1`.

**Revisit when:** a sweep wastes >20% of GPU-hours on clearly-bad
trials. Don't preempt the pain (Lesson 4: don't refactor before pain).

### Static-HTML dashboard (parent Lesson 5)

If `plans show` outgrows the CLI table, render jinja+plotly to
`${RUN_ROOT}/plans/<plan_id>/dashboard.html`. No service. TUI direction
dropped.

**Revisit when:** filterable trial views or parallel-coordinate plots
become a recurring ask in `gx plans show` workflow.

### Drop `Plan.plan_args` field

Probably redundant with `tags.graphids.plan_args` on MLflow runs.

**Revisit when:** next chassis cleanup pass — leave the field today
to avoid touching the schema mid-flight.

---

## Explicitly not doing

- `graphids launch X --cluster Y` (render+submit fused) — violates
  Invariant 1.
- DAG runner / `depends_on_row_name` on rows — even "advisory" is the
  on-ramp (parent Lesson 6, Invariant 2 + chassis-invariants §"DAG
  runners").
- TUI for plan management — Lesson 5 dropped it.
- Replacing MLflow with Aim/ClearML — locked by infra.
- Hand-rolled JSON-IPC layer — Pydantic at the JSON boundary pays.

---

## Risks for the held items

1. **`--filter` partial-sweep silent gap.** Mitigation: `plans show`
   reports rows in the plan_module's full render with no MLflow run yet.
2. **MLflow tag drift** (rename of `graphids.plan_args` etc.) silently
   breaks the reproduction contract. Mitigation: tag-name list lives
   in `_mlflow.IDENTITY_KEYS`; `plans show` warns on missing tags.
