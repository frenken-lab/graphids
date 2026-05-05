# GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection via a 3-stage knowledge distillation chain:
VGAE (unsupervised reconstruction) → GAT (supervised classification) →
fusion. Large models compress into small models via KD auxiliaries for
edge deployment. Stages are trained as independent ablation rows;
multi-stage pipelines (Python plans under `graphids/configs/plans/`)
render to a JSON blueprint via `graphids run` — the user/LLM walks the
rows and invokes `graphids exec` (in-process) or `graphids submit`
(SLURM) per row. No pipeline driver. See
`.claude/rules/single-submission-primitive.md`.

## Key Commands

```bash
# Four-step chassis: render → blueprint → exec → submit.
# Step 1+2: import a Python plan, validate as a Blueprint, write JSON array.
# `<plan>` is a dotted module under graphids.configs.plans.
python -m graphids run ofat --dataset hcrl_sa --seed 42 -o plan.json

# Step 3: execute one row in-process (login-node smoke / non-SLURM).
jq -c '.[0]' plan.json | xargs -I{} python -m graphids exec --row {}
echo '<row-json>' | python -m graphids exec --row -

# Step 4: submit one row to SLURM via Parsl. Prints jid on stdout.
jq -c '.[]' plan.json | while read row; do
    python -m graphids submit --row "$row" --cluster pitzer --length long
done
# Same-batch dependency chain (afterok = data dep, afterany = preempt-resume):
python -m graphids submit --row "$row" --cluster cardinal \
    --depends-on-afterok 12345 --ckpt-path /path/to/upstream/best.ckpt

# Per-checkpoint artifacts (CKA / embeddings / loss landscape / fusion policy)
# are an `analyze` blueprint action — author a plan under
# graphids/configs/plans/ops/ emitting one AnalyzeRow per checkpoint,
# then run/exec/submit like any row:
python -m graphids run ops.analyze_gat --dataset hcrl_sa --seed 42 -o analyze.json
jq -c '.[0]' analyze.json | xargs -I{} python -m graphids exec --row {}
```

## CLI Architecture

The four user-facing primitives are pure stages. Each does exactly one
thing and feeds the next; no stage submits, queries MLflow, or
orchestrates multiple jobs.

| Stage  | Command                                | Module                                   | What it does                                                                                           |
| ------ | -------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| render | `graphids run <plan>`                  | `graphids/cli/commands.py`               | Imports `graphids.configs.plans.<plan>`, calls `build(dataset, seed)`, validates as `BlueprintArray`, writes JSON. |
| exec   | `graphids exec --row <json>`           | `graphids/cli/commands.py`               | Executes one row in-process via `graphids.orchestrate.run_row`. Dispatches on `row.action` (fit/test/extract/analyze). |
| submit | `graphids submit --row <json>`         | `graphids/cli/commands.py`               | Submits one row to SLURM via Parsl `SlurmProvider`. Prints jid.                                        |
| ops    | `analyze` / `cmd` / `extract` rows     | `graphids/configs/plans/ops/*.py`        | Ops are rows too — analyze ckpt, HF push, cache rebuild. Same run/exec/submit chassis.                 |

`graphids/__main__.py` imports each submodule to register Typer commands.
`app.py` owns the root app + shared option types.

**Config resolution** — Single path: a Python plan module under
`graphids/configs/plans/` exposes `build(dataset, seed) -> list[dict]`;
`BlueprintArray` / `TrainRow` (`graphids/blueprint.py`) validates it.
`run_row` walks nested `class_path` blocks and instantiates via
importlib with signature-filtered kwargs. Loss fragments are true
`{class_path, init_args}` blocks — no `inject_loss_fn` helper.
Composers (`graphids/configs/compose/`) return a frozen `RowSpec`
whose `rendered` is a locked `ml_collections.ConfigDict` (typo'd field
access raises with a did-you-mean hint). See
`docs/reference/config-architecture.md`.

**SLURM submission** — `graphids submit` is the ONLY caller of
`SlurmProvider.submit`. The sbatch script carries the literal command
`python -m graphids exec --row '<json>' [--ckpt-path X]` — no pickle,
no stale-pickle bug. Profiles in `configs/resources/submit_profiles.json`
keyed `[mode][cluster][length]` translate to Parsl `SlurmProvider`
kwargs (partition, cores, mem, walltime, gpus, signal-delay).
`SrunLauncher` wraps the command. Preempt-resume kept via SIGUSR2 trap
in `graphids/runtime.py` that re-submits the row with `--ckpt-path
last.ckpt` and `--dependency=afterany:$SLURM_JOB_ID`.

Library entrypoint: `graphids.slurm.submit_row(row, cluster=..., ...)`.
See `.claude/rules/slurm-hpc.md`.

Fusion uses a single `configs/models/fusion/` module that dispatches on
the `fusion_method` TLA over the method libsonnets.

## Lake-root data layout (current state, 2026-05-04)

`/fs/ess/PAS1266/graphids/` contains only `dev/` (per-user run roots).
No `raw/`, no `cache/`, no `mlflow.db`. Any training submission against
`hcrl_sa` will die in `dataset.build()` →
`vocab.scan_arb_ids("/fs/ess/PAS1266/graphids/raw/hcrl_sa/train_01_attack_free")`
with `FileNotFoundError`. Restore the raw CSVs (or repoint the catalog)
before submitting.

Example failure (jid 47282855, ops.gat_taunorm_smoke): refactor paths
all clean — Pydantic round-trip, `_ensure_runtime`, class-side
`_SCALES`, `compose.callbacks_base` ran fine; failure was downstream
data, not config.

## Rules (auto-loaded from `.claude/rules/`)
