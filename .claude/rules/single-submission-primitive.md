# Single Submission Primitive

> Four pure stages: **render → blueprint → exec → submit**. Pipelines are
> JSON arrays, not runners. Never add a Python pipeline driver, never emit
> an executable artifact (bash script, etc.) that orchestrates jobs in bulk.

## The architecture

| Concern | Tool | What it does |
|---|---|---|
| Render plan | `graphids run <plan.jsonnet> -o plan.json` | Renders + validates as `BlueprintArray`, writes JSON. No submit, no MLflow query. |
| Execute one row | `graphids exec --row <json>` | Calls `orchestrate.run_row(TrainRow)`. Login-node smoke / non-SLURM. Dispatches on `row.action` (fit/test). |
| Submit one row | `graphids submit --row <json> --cluster <c> [--length L]` | Atomic Parsl `SlurmProvider.submit`. The ONLY caller of `submit_row`. Returns jid on stdout. |
| Same-batch deps | `--depends-on-afterok <jid>` (data dep) / `--depends-on-afterany <jid>` (preempt-resume chain) | Adds `#SBATCH --dependency=...`. |

The sbatch script carries the literal command
`python -m graphids exec --row '<json>' [--ckpt-path X]` — no pickle,
no stale-pickle bug.

## What this rule blocks

- **Pipeline drivers.** No `run-ofat` command, no `OFAT_DAG.execute()`,
  no Python loop walking plan rows calling `submit_row()`.
- **Executable artifacts.** `graphids run` outputs JSON (data), not bash
  (action). Iteration is the user/LLM's job:
  ```
  jq -c '.[]' plan.json | while read row; do
      graphids submit --row "$row" --cluster pitzer
  done
  ```
- **Per-pipeline submission flags.** No `--ofat-mode`, no `--sweep-strategy`.
  New pipeline types add a `configs/plans/<name>.jsonnet`, nothing else.
  New declarative state goes on `TrainRow` / `BlueprintArray`
  (`graphids/blueprint.py`).
- **Multi-job entry points.** No `submit-many`, no `submit-batch`. N jobs =
  N invocations of `graphids submit`.
- **Status command resurrection.** Use MLflow's UI or
  `_mlflow.build_search_filter` — don't re-introduce `graphids status`.

## What this rule allows

- New plan jsonnets under `configs/plans/`. Produce new rows; `graphids run`
  consumes them with no code changes.
- New `TrainRow` / `BlueprintArray` fields when a plan needs new state.
- New cluster profiles in `configs/resources/submit_profiles.json`.

## Why one primitive

Past failures: `OFAT_DAG` Python orchestrator (~470 LOC, removed c5873de),
`launch.ablation` helpers (split logic across CLI + driver), bash artifacts
from `slurm/run.py:render_plan_script` (looked like data, behaved like
action — removed 2026-04-27), submitit pickle path (code fixes didn't reach
pending jobs — replaced 2026-05-01 with Parsl + literal exec command in
sbatch). Each was Claude reaching for "let's add a wrapper that does the
orchestration." The cost is path proliferation.

## Decision rule

If you're tempted to add something that:
- iterates over `(plan_row, dataset, seed)` calling `submit_row()` per row, OR
- produces a file that, when executed, submits multiple jobs, OR
- adds a flag to `submit` keyed on pipeline-level context, OR
- re-introduces a `graphids status`-style read-and-act helper

→ **Stop.** The right answer is a new plan jsonnet + the user/LLM iterating
over `graphids run` JSON output.
