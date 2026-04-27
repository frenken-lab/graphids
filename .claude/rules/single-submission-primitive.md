# Single Submission Primitive

> One job submission primitive: `graphids submit`. Pipelines are
> blueprints (JSONL), not runners. Never add a Python pipeline driver,
> never emit an executable artifact (bash, shell script, anything that
> orchestrates jobs in bulk).

## The architecture

| Concern | Tool | What it does |
|---|---|---|
| Submit one job | `graphids submit <preset.jsonnet>` | Atomic submitit call. Returns jid. The ONLY place that calls `submitit.AutoExecutor.submit`. |
| Pipeline blueprint | `graphids run <plan.jsonnet>` | Renders the plan to **JSONL on stdout** — one row per node, with `submit_command` strings. Does not submit anything. Does not write an executable file. |
| Plan status | `graphids status <plan.jsonnet>` | Read-only MLflow query per node. Renders a table or JSON. Does not submit. |
| Same-batch deps | `graphids submit ... --depends-on <variant>:<seed>` | One flag dispatches: FINISHED upstream → inject ckpt as TLA; RUNNING upstream → afterok on its `slurm.slurm_job_id` MLflow tag; missing → hard error. |

## What this rule blocks

- **Pipeline drivers.** No `python -m graphids run-ofat`, no
  `slurm/sweep_runner.py`, no `OFAT_DAG.execute()`, no Python loop
  that walks plan nodes calling `submit()`. The 2026-04-23 collapse
  removed `OFAT_DAG`; this rule prevents Claude from re-introducing the
  same shape under a new name.
- **Executable artifacts.** `graphids run` outputs JSONL (data), not
  bash (action). The 2026-04-27 deletion removed the bash renderer; do
  not bring it back. Workflow:
  ```
  graphids run plan.jsonnet --dataset X --seed N --cluster C \
      | jq -r '.submit_command' \
      | while read cmd; do eval "$cmd"; done   # user/LLM choice, not generated
  ```
  The user (or an LLM walking the JSONL) decides how to iterate. The
  graphids codebase does not own that loop.
- **Per-pipeline submission flags.** No `--ofat-mode`, no
  `--sweep-strategy`, no `--curriculum-driver`. Every new pipeline
  type adds a `configs/plans/<name>.jsonnet`, nothing else. If a
  plan needs new declarative fields, add them to `slurm/dag.py:Node`
  with pydantic — never to the CLI surface.
- **Multi-job entry points.** No second `submit-many` or `submit-batch`
  command. If the user wants to submit N jobs, that's N invocations of
  `graphids submit`.

## What this rule allows

- New plan jsonnets under `configs/plans/`. They produce new JSONL
  rows; `graphids run` and `graphids status` work on them with no code
  changes.
- New ablation presets. They become rows in some plan; the same
  `submit` primitive launches them.
- New variants in `DEPENDS_ON_TLA` (`slurm/dependencies.py`). The
  registry is the right place for producer→consumer-TLA mapping.
- Read-only views of the blueprint (`graphids status` already does
  this). Adding a new view is fine as long as it doesn't submit.

## Why one primitive

Three failure modes the previous architecture had:

1. **`OFAT_DAG` Python class** with hardcoded stage tuples and
   in-process orchestration (~470 LOC). Removed in c5873de. Each
   pipeline type would have wanted its own.
2. **`launch.ablation`** + **`_build_tlas`** helpers split the submit
   logic between CLI and pipeline driver. Two paths through every
   change. Removed in c5873de.
3. **Executable bash artifact** from `slurm/run.py:render_plan_script`.
   Looked like data, behaved like action. `set -euo pipefail` made it
   all-or-nothing; partial re-run was awkward; jids were captured via
   `$(...)` with no failure surface. Removed 2026-04-27.

Each was Claude reaching for "let's add a wrapper that does the orchestration."
The cost is path proliferation: every refactor has to update N call
sites, and the divergence between paths becomes the source of bugs
(2026-04-24 jsonnet/Python `RUN_ROOT` drift was one of these).

## Decision rule for future Claude

If you're tempted to add a function/CLI/script that:

- iterates over multiple `(preset, dataset, seed)` combinations and
  calls `submit()` for each, OR
- produces a file that, when executed, submits multiple jobs, OR
- adds a flag to `submit` that changes its behavior based on
  pipeline-level context (sweep ID, ablation group, etc.)

→ **Stop.** The right answer is a new plan jsonnet + the existing
`graphids run` JSONL output + the user/LLM iterating over rows. If
that workflow has a real gap, write it down as an issue first; do not
patch around it with a new path.
