# Single Submission Primitive

> Four pure stages: **render → blueprint → exec → submit**. Pipelines are
> JSON arrays, not runners. Never add a Python pipeline driver, never emit
> an executable artifact (bash, shell script, anything that orchestrates
> jobs in bulk).

## The architecture

| Concern | Tool | What it does |
|---|---|---|
| Render plan | `graphids run <plan.jsonnet> -o plan.json` | Renders plan jsonnet, validates as `BlueprintArray`, writes JSON array. No submit, no MLflow query. |
| Execute one row in-process | `graphids exec --row <json>` | Calls `graphids.orchestrate.run_row(TrainRow)`. Login-node smoke / non-SLURM path. Dispatches on `row.action` (fit/test). |
| Submit one row to SLURM | `graphids submit --row <json> --cluster <c> [--length L]` | Atomic Parsl `SlurmProvider.submit` call. The ONLY caller of `submit_row`. Returns jid on stdout. |
| Same-batch deps | `graphids submit --row <json> --depends-on-afterok <jid>` <br> `graphids submit --row <json> --depends-on-afterany <jid>` | Adds `#SBATCH --dependency=after{ok,any}:<jid>`. `afterok` for data deps, `afterany` for preempt-resume chains. |

The sbatch script carries the literal command
`python -m graphids exec --row '<json>' [--ckpt-path X]` — no pickle,
no stale-pickle bug.

## What this rule blocks

- **Pipeline drivers.** No `python -m graphids run-ofat`, no
  `slurm/sweep_runner.py`, no `OFAT_DAG.execute()`, no Python loop that
  walks plan rows calling `submit_row()`. The 2026-04-23 collapse removed
  `OFAT_DAG`; the 2026-05-01 four-step rebuild deleted `slurm/dag.py`,
  `slurm/dependencies.py`, `slurm/sizing.py`, `slurm/run.py`,
  `slurm/status.py`, `cli/training.py`, and `cli/compare.py`. This rule
  prevents Claude from re-introducing the same shape under a new name.
- **Executable artifacts.** `graphids run` outputs JSON (data), not bash
  (action). The 2026-04-27 deletion removed the bash renderer; do not
  bring it back. Workflow:
  ```
  graphids run plan.jsonnet --dataset X --seed N -o plan.json
  jq -c '.[]' plan.json | while read row; do
      graphids submit --row "$row" --cluster pitzer  # user/LLM choice, not generated
  done
  ```
  The user (or an LLM walking the array) decides how to iterate. The
  graphids codebase does not own that loop.
- **Per-pipeline submission flags.** No `--ofat-mode`, no
  `--sweep-strategy`, no `--curriculum-driver`. Every new pipeline type
  adds a `configs/plans/<name>.jsonnet`, nothing else. If a plan needs
  new declarative fields, add them to `graphids/blueprint.py` (pydantic
  `TrainRow` / `BlueprintArray`) — never to the CLI surface.
- **Multi-job entry points.** No `submit-many`, no `submit-batch`, no
  `graphids status` query helper. If the user wants to submit N jobs,
  that's N invocations of `graphids submit`. If they want status, they
  query MLflow directly (or via a read-only ops command — but it must
  not also submit).
- **Status command resurrection.** `graphids status` was a thin wrapper
  over `mlflow.search_runs` keyed by plan blueprint. It got deleted in
  the 2026-05-01 chassis rebuild. Do not bring it back unless the user
  explicitly asks; MLflow's own UI + `_mlflow.build_search_filter` cover
  the use case.

## What this rule allows

- New plan jsonnets under `configs/plans/`. They produce new rows;
  `graphids run` consumes them with no code changes.
- New ablation rows. They become entries in some plan's array; the same
  `submit` primitive launches them.
- New `TrainRow` / `BlueprintArray` fields in `graphids/blueprint.py`
  when a plan needs new declarative state. Pydantic owns the schema.
- New cluster profiles in `configs/resources/submit_profiles.json`
  (`[mode][cluster][length]`).

## Why one primitive

Failure modes the previous architectures had:

1. **`OFAT_DAG` Python class** with hardcoded stage tuples and in-process
   orchestration (~470 LOC). Removed in c5873de. Each pipeline type
   would have wanted its own.
2. **`launch.ablation` + `_build_tlas`** helpers split the submit logic
   between CLI and pipeline driver. Two paths through every change.
   Removed in c5873de.
3. **Executable bash artifact** from `slurm/run.py:render_plan_script`.
   Looked like data, behaved like action. `set -euo pipefail` made it
   all-or-nothing; partial re-run was awkward; jids were captured via
   `$(...)` with no failure surface. Removed 2026-04-27.
4. **submitit pickle path** — `_TrainingJob.checkpoint()` + pickled
   closures meant code fixes didn't reach pending jobs. Replaced
   2026-05-01 with Parsl + literal `python -m graphids exec --row '...'`
   in the sbatch script. The pickle hazard is now structurally impossible.

Each was Claude reaching for "let's add a wrapper that does the
orchestration." The cost is path proliferation: every refactor has to
update N call sites, and the divergence between paths becomes the source
of bugs (2026-04-24 jsonnet/Python `RUN_ROOT` drift was one of these).

## Decision rule for future Claude

If you're tempted to add a function/CLI/script that:

- iterates over multiple `(plan_row, dataset, seed)` combinations and
  calls `submit_row()` for each, OR
- produces a file that, when executed, submits multiple jobs, OR
- adds a flag to `submit` that changes its behavior based on
  pipeline-level context (sweep ID, ablation group, etc.), OR
- re-introduces a `graphids status` style read-and-act helper

→ **Stop.** The right answer is a new plan jsonnet + the existing
`graphids run` JSON output + the user/LLM iterating over rows. If that
workflow has a real gap, write it down as an issue first; do not patch
around it with a new path.
