# Submit flow — atomic submission, end to end

> Status: **implemented** | Companions: `orchestration.md`,
> `observability.md`, `config-architecture.md`,
> `.claude/rules/single-submission-primitive.md`

How a job goes from CLI invocation on the login node to a running
training process on a compute node, including preemption recovery.

## The chassis

Every job is one row, submitted atomically:

```bash
# 1. Render a Python plan to a row array.
graphids run <plan-module> --dataset X --seed N -o plan.json

# 2. Iterate — the user/LLM is the loop. There is NO Python pipeline driver.
jq -c '.[]' plan.json | while read row; do
    graphids submit --row "$row" --cluster pitzer --length long
done
```

`<plan-module>` is a dotted name under `graphids.configs.plans`
(e.g. `supervised`, `ofat`, `ops.gat_taunorm_smoke`). See
`config-architecture.md`.

## Atomic: `graphids submit --row '<json>'`

Login-node phase (until `sbatch` returns):

1. **`graphids/cli/commands.py:submit_cli`** parses Typer flags, loads
   the row JSON via `BlueprintArray.model_validate([json.loads(...)])[0]`
   (validates the row's discriminated-union shape).
2. **`graphids/slurm/submit.py:submit_row(row, cluster, length, ...)`**
   is the one library entrypoint and the ONLY caller of Parsl's
   `SlurmProvider.submit`. Looks up
   `_PROFILES[mode][cluster][length]` from
   `configs/resources/submit_profiles.json` (raw `SlurmProvider`
   kwargs — `**profile` splat, no parser layer). `signal_delay_s` is
   the one extension: becomes `#SBATCH --signal=USR2@N`.
3. **The sbatch script body is a literal bash string**:
   ```
   python -m graphids exec --row '<row JSON>' [--ckpt-path X]
   ```
   No pickle of Python closures. Code edits committed *after* a job
   queues DO reach it — the job re-imports current source at exec time.
4. **`SrunLauncher` wraps the command.** `--dependency=afterok:<jid>`
   (or `afterany`) is set when `--depends-on-afterok` /
   `--depends-on-afterany` is passed. Returns the jid; CLI prints it.

Compute-node phase:

5. **`srun python -m graphids exec --row '<json>'`** runs in the
   allocated resources. `_load_row` revalidates via `BlueprintArray`,
   then `orchestrate.run_row(row, ckpt_path=...)` dispatches on
   `row.action`.
6. For `fit` / `test`: `_instantiate` walks the `class_path` tree,
   `_mlflow.start_training_run` opens the run (status-gated resume),
   `trainer.fit(...)` or `trainer.test(...)` runs.
   `MLflowTrainingCallback` forwards per-epoch metrics; `on_fit_end`
   closes FINISHED.

## Preemption auto-resume

Profiles set `signal_delay_s=300` → `--signal=USR2@300` (USR2 because
NCCL catches USR1). Five minutes before walltime, Lightning's
`SLURMEnvironment(auto_requeue=True, requeue_signal=SIGUSR2)` plugin
(wired by `orchestrate._trainer_kwargs`) catches the signal and calls
`scontrol requeue $SLURM_JOB_ID` — same job ID, downstream `afterok`
chains stay valid. The replacement run picks up from
`{run_dir}/checkpoints/last.ckpt` without manual intervention.

## Same-batch dependency chains

For multi-stage rows in the same plan (e.g. fusion needs an extracted
states cache), thread dependencies via the submit flag:

```bash
EXTRACT_JID=$(jq -c '.[0]' plan.json | xargs -I{} \
    graphids submit --row {} --cluster pitzer)
for r in $(jq -c '.[1:][]' plan.json); do
    graphids submit --row "$r" --cluster pitzer \
        --depends-on-afterok "$EXTRACT_JID"
done
```

`afterok` for data dependencies (downstream waits for upstream FINISH);
`afterany` for preempt-resume chains (downstream runs whether upstream
exited cleanly or was preempted).

## Ckpt flag semantics

| Flag | Semantics | Wired to |
|---|---|---|
| `--ckpt-path` | Resume the *current* row from a ckpt — passed to `orchestrate.run_row(row, ckpt_path=...)`. Used by preempt-resume. | `submit_row(..., ckpt_path=...)` → sbatch command suffix. |
| `--depends-on-afterok <jid>` | Add `#SBATCH --dependency=afterok:<jid>` (data dep). | `SlurmProvider` dependency string. |
| `--depends-on-afterany <jid>` | Add `#SBATCH --dependency=afterany:<jid>` (preempt-resume chain). | Same. |

Upstream-teacher ckpt paths (e.g. VGAE → GAT distillation) flow through
the row's `upstreams` array, populated at plan-build time via
`graphids.configs.catalog.best_ckpt(...)`. The row carries everything
the compute node needs; no MLflow lookup at submit time, no
`--depends-on V[:S]` flag.

## File map

| File | Role |
|---|---|
| `graphids/cli/commands.py` | `run` / `exec` / `submit` Typer wrappers. |
| `graphids/configs/blueprint.py` | `BlueprintArray`, `TrainRow`, `ExtractRow`, `AnalyzeRow`, `CmdRow`, `RenderedConfig`, `ClassPath`, `TrainerCfg` — discriminated-union row schema + typed rendered-config schema. |
| `graphids/orchestrate.py` | `run_row(row)` dispatch, `_instantiate` recursion, `UpstreamLineageCallback`, `_ensure_runtime` (spawn + structlog). Preempt-resume via Lightning's `SLURMEnvironment` plugin. |
| `graphids/slurm/submit.py` | `submit_row(...)` library entrypoint; one Parsl `SlurmProvider.submit` site. |
| `configs/resources/submit_profiles.json` | Raw Parsl `SlurmProvider` kwargs keyed `[mode][cluster][length]`. |
| `graphids/configs/plans/<name>.py` | Plan modules — `build(dataset, seed) → list[dict]`. |

## Non-obvious invariants

- **No pickle.** The sbatch script carries a literal command string;
  `graphids/` source on the compute node is whatever's checked out at
  exec time. Code fixes between submit and exec DO reach the job.
- **MLflow is a hard dep.** Failures from `start_training_run` /
  `log_test_run` propagate. The two soft-failure paths
  (`log_params` resume conflict, `end_training_run` cleanup) are
  documented in `_mlflow.py`'s module docstring.
- **`run_dir` is computed in the plan**, baked into the row's
  `identity`, and consumed by the compute node verbatim. No
  re-computation, no scheduler re-query.
- **`graphids submit` is the ONLY caller of `submit_row`.** No
  `submit-many`, no `submit-batch`, no Python loop calling
  `submit_row()` per row. N jobs = N invocations of `graphids submit`.
- **There is no `graphids status`.** Use the MLflow UI or
  `_mlflow.build_search_filter(...)` for cross-run queries.
