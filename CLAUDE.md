# GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection via a 3-stage knowledge distillation chain:
VGAE (unsupervised reconstruction) → GAT (supervised classification) →
fusion. Large models compress into small models via KD auxiliaries for
edge deployment. Stages are trained as independent ablation rows;
multi-stage pipelines (Python plans under `graphids/plan/plans/`)
render to a JSON blueprint via `graphids run` — the user/LLM walks the
rows and invokes `graphids exec` (in-process) or `graphids submit`
(SLURM) per row. No pipeline driver. See
`.claude/rules/chassis-invariants.md`.

## Key Commands

The CLI is `graphids` or its short alias `gx` (registered as a console
script in `pyproject.toml` — both invoke `graphids.__main__:main`).
Examples below use `gx`.

```bash
# Discover what plans exist + what they'd render
gx plans available
gx plans describe ablations.gat_loss -d hcrl_sa -s 42

# Render a plan to JSON. Pydantic validates structure; typos raise here.
gx run ablations.gat_loss --dataset hcrl_sa --seed 42 -o plan.json
gx run ablations.gat_loss -d hcrl_sa -s 42 --filter 'focal*' -o plan.json

# Smoke one row in-process (no SLURM). Imports torch.
gx exec --plan plan.json --row-name focal_fit
gx exec --row "$(jq -c '.rows[0]' plan.json)"

# Submit one row to SLURM via Parsl. Prints jid on stdout.
gx submit --plan plan.json --row-name focal_fit --cluster pitzer
# Same-batch dependency chain (afterok = data dep, afterany = preempt-resume):
gx submit --plan plan.json --row-name focal_test --cluster pitzer \
    --depends-on-afterok 12345 --ckpt-path /path/to/best.ckpt

# Submit MANY rows from a rendered plan. MLflow-aware filtering.
gx plans submit --plan plan.json -C pitzer
gx plans submit --plan plan.json -C pitzer --resume        # skip FINISHED
gx plans submit --plan plan.json -C pitzer --filter '*-test' --dry-run

# Monitor + evaluate
gx q                                       # squeue all clusters
gx qpend                                   # pending jobs + reason
gx qhist                                   # sacct since today
gx nodes                                   # sinfo gpu/cpu partitions
gx disk                                    # du on $RUN_ROOT + scratch
gx plans show <plan_id>                    # consolidated MLflow table for one plan
gx plans show <plan_id> --status FAILED --names-only   # machine-readable
gx plans where <plan_id> --row focal_fit   # run_dir / ckpt / stderr / mlflow run

# Per-checkpoint artifacts (CKA / embeddings / loss landscape / fusion policy)
# are an `analyze` action — author a plan emitting AnalyzeRow per checkpoint:
gx run ops.analyze_gat -d hcrl_sa -s 42 -o analyze.json
gx plans submit --plan analyze.json -C pitzer
```

## CLI Architecture

Render is pure JSON. Submit (single-row or `plans submit` multi-row)
consumes the JSON. MLflow is the trial-state store. Row JSON is frozen
in the sbatch script — drift resistance is architectural. See
`.claude/rules/chassis-invariants.md` for the four properties this
preserves.

| Stage  | Command                                | Module                                   | What it does                                                                                           |
| ------ | -------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| render | `graphids run <plan>`                  | `graphids/cli/commands.py`               | Imports `graphids.plan.plans.<plan>`, calls `build(dataset, seed)`, validates as `Plan`, writes JSON. `--filter <glob>` subsets rows. |
| exec   | `graphids exec --row <json>` or `--plan FILE --row-name NAME` | `graphids/cli/commands.py`               | Executes one row in-process via `graphids.orchestrate.run_row`. Dispatches on `row.action` (fit/test/extract/analyze/cache). |
| submit | `graphids submit --row <json>` or `--plan FILE --row-name NAME` | `graphids/cli/commands.py`               | Submits one row to SLURM via Parsl `SlurmProvider`. Prints jid.                                        |
| bulk   | `graphids plans submit --plan FILE --cluster X` | `graphids/cli/plans.py`                  | Walks rendered plan, submits rows. `--filter`/`--resume`/`--dry-run`. Each row's outcome is its own log line. |
| views  | `graphids plans {list,available,describe,show,where}` | `graphids/cli/plans.py`                  | Read-only MLflow + filesystem queries.                                                                 |
| ops    | `analyze` / `cache` / `extract` rows   | `graphids/plan/plans/{smoke,data,ablations}/*.py` | Ops are rows too — analyze ckpt, HF push, cache rebuild. Same run / submit chassis.           |

`graphids/__main__.py` imports each submodule to register Typer commands.
`app.py` owns the root app + shared option types.

**Config resolution** — Single path: a Python plan module under
`graphids/plan/plans/` exposes `build(dataset, seed) -> list[dict]`;
`Plan` / `TrainRow` (`graphids/plan/schema.py`) validates it.
`run_row` walks nested `class_path` blocks and instantiates via
importlib with signature-filtered kwargs. Loss fragments are true
`{class_path, init_args}` blocks — no `inject_loss_fn` helper.
Composers (in `graphids/plan/compose.py`) return a frozen `RowSpec`
whose `rendered` is a locked `ml_collections.ConfigDict` (typo'd field
access raises with a did-you-mean hint). See
`docs/reference/config-architecture.md`.

**SLURM submission** — `graphids.slurm.submit.submit_row` is the ONLY
caller of `SlurmProvider.submit`. Both `graphids submit` (single row)
and `graphids plans submit` (multi-row, MLflow-aware) ultimately call
it. The sbatch script carries the literal command
`python -m graphids exec --row '<json>' [--ckpt-path X]` — no pickle,
no stale-pickle bug. Row JSON is frozen here at submit time; this is
the architectural drift-resistance property (see
`chassis-invariants.md`). Profiles in `configs/resources/submit_profiles.json`
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
