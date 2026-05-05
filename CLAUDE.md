# GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection via a 3-stage knowledge distillation chain:
VGAE (unsupervised reconstruction) → GAT (supervised classification) →
fusion. Large models compress into small models via KD auxiliaries for
edge deployment. Stages are trained as independent ablation rows;
multi-stage pipelines (Python plans under `graphids/plan/plans/`)
render to a JSON blueprint via `graphids run` — the user/LLM walks the
rows and invokes `graphids exec` (in-process) or `graphids submit`
(SLURM) per row. No pipeline driver. See
`.claude/rules/single-submission-primitive.md`.

## Key Commands

```bash
# Four-step chassis: render → blueprint → exec → submit.
# Step 1+2: import a Python plan, validate as a Blueprint, write JSON array.
# `<plan>` is a dotted module under graphids.plan.plans.
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
# graphids/plan/plans/{smoke,data}/ emitting one AnalyzeRow per checkpoint,
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
| render | `graphids run <plan>`                  | `graphids/cli/commands.py`               | Imports `graphids.plan.plans.<plan>`, calls `build(dataset, seed)`, validates as `BlueprintArray`, writes JSON. |
| exec   | `graphids exec --row <json>`           | `graphids/cli/commands.py`               | Executes one row in-process via `graphids.orchestrate.run_row`. Dispatches on `row.action` (fit/test/extract/analyze). |
| submit | `graphids submit --row <json>`         | `graphids/cli/commands.py`               | Submits one row to SLURM via Parsl `SlurmProvider`. Prints jid.                                        |
| ops    | `analyze` / `cache` / `extract` rows   | `graphids/plan/plans/{smoke,data,ablations}/*.py` | Ops are rows too — analyze ckpt, HF push, cache rebuild. Same run/exec/submit chassis.        |

`graphids/__main__.py` imports each submodule to register Typer commands.
`app.py` owns the root app + shared option types.

**Config resolution** — Single path: a Python plan module under
`graphids/plan/plans/` exposes `build(dataset, seed) -> list[dict]`;
`BlueprintArray` / `TrainRow` (`graphids/plan/blueprint.py`) validates it.
`run_row` walks nested `class_path` blocks and instantiates via
importlib with signature-filtered kwargs. Loss fragments are true
`{class_path, init_args}` blocks — no `inject_loss_fn` helper.
Composers (in `graphids/plan/compose.py`) return a frozen `RowSpec`
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
wired in `graphids/orchestrate.py` (Lightning `SLURMEnvironment(auto_requeue=True,
requeue_signal=SIGUSR2)`), which calls `scontrol requeue` — same job ID,
downstream `afterok` deps stay valid.

Library entrypoint: `graphids.slurm.submit_row(row, cluster=..., ...)`.
See `.claude/rules/slurm-hpc.md`.

Fusion lives at `graphids/core/models/fusion/`; dispatch on the
`fusion_method` field happens inside the composer.

## Lake-root data layout (current state, 2026-05-05)

`/fs/ess/PAS1266/graphids/` holds:
- `dev/` — per-user run roots
- `raw/can-train-and-test-v1.5/` — Lampe et al. dataset (cloned from
  https://bitbucket.org/brooke-lampe/can-train-and-test-v1.5). Contains
  `hcrl_sa/`, `hcrl_ch/`, `set_01..set_04/`, `helper/`. Catalog paths
  expect underscored names (e.g. `hcrl_sa/train_01_attack_free`).
- `cache/v{PREPROCESSING_VERSION}/{dataset}/voc_{scope}/` — built on
  first `dataset.build()` call (or via the `data.rebuild_cache` plan).

## Rules (auto-loaded from `.claude/rules/`)
