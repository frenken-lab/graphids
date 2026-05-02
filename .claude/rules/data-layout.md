# GraphIDS Data Layout — Store Ownership

**Three roots — keep them straight.** `LAKE_ROOT` (shared, cross-user)
holds metadata + caches that everyone reads. `RUN_ROOT` (per-user) holds
this user's experiment writes. `run_dir` is the per-(variant,seed) leaf
under `RUN_ROOT`. Conflating `LAKE_ROOT` and `RUN_ROOT` is what produced
the 2026-04-24 drift between Python settings and the (now-deleted)
jsonnet TLA defaults — that bug is impossible by construction now.

> Filesystem tree + per-file write-path inventory:
> `docs/reference/write-paths.md`. This file owns the **store-ownership
> table** ("for signal X, query where?") and the rules that prevent
> duplicating stores.

## Store ownership — don't duplicate

| Signal | Where | Reader |
|---|---|---|
| Run metadata (params, tags, timestamps) | `mlflow.db` | `mlflow.search_runs` + `graphids._mlflow.build_search_filter(...)` |
| Per-epoch scalar metrics (train/val_loss, lr) | `mlflow.db` | `client.get_metric_history(run_id, key)` |
| Device telemetry (GPU util, VRAM, CPU, mem) | `mlflow.db` (system-metrics sampler, 5s) | MLflow UI charts |
| Final test metrics | `mlflow.db` (test-phase row) | `search_runs` filter `tags.graphids.phase = 'test'` |
| Dataset identity (cache digest) | `mlflow.db` `Dataset` entity | `search_runs` filter `dataset.digest = '...'`; MLflow UI "Used Datasets" panel |
| LoggedModel (ckpt metadata entity, upstream lineage handle) | `mlflow.db` (MLflow 3) | `search_logged_models(experiment_ids=[...], filter_string="source_run_id = '...'")` |
| Checkpoint bytes | `{run_dir}/checkpoints/` | `torch.load` via `_fs.atomic_load` |
| Checkpoint SHA256 | `.sha256` sidecar + `LoggedModel.tags.graphids.ckpt_sha256` | `atomic_load` verifies; LoggedModel tag is provenance |
| Span lifecycle + log events | `{run_dir}/traces.jsonl` | `jq` / manual grep for debugging |
| Validated jsonnet config | `{run_dir}/resolved.json` | replay / debugging |

## Key rules

1. **Never `mlflow.log_artifact` or `mlflow.pytorch.log_model`.** Checkpoints
   stay as filesystem paths so resume, KD-student teacher loading, and fusion
   upstream-ckpt flow all work via direct paths. MLflow rows link to them via
   the `graphids.run_dir` tag. MLflow-3 `LoggedModel` (metadata-only via
   `MlflowClient.create_logged_model`) is OK — it takes no artifact bytes.
   `mlflow.data.Dataset` / `mlflow.log_input` is OK for the same reason.

2. **Two MLflow rows per run under the same `run_name`.** Fit phase (`tags.graphids.phase = 'fit'`)
   and test phase (`tags.graphids.phase = 'test'`). `run_name` is deterministic:
   `{group}_{variant}_{dataset}_seed{N}[_{cluster}]`. Filter by the phase tag
   when `search_runs` returns what looks like duplicates. **Fit is resumable**
   (status-gated: FAILED/KILLED → resume same `run_id`); **test is always-fresh**
   (new `run_id` each invocation; `compare.py` dedups to latest FINISHED per
   (variant, seed)).

3. **MLflow run lifecycle spans `stage.train` + the callback.**
   `_mlflow.start_training_run` opens the run in `stage.train::train` before
   `trainer.fit`; `MLflowTrainingCallback.on_fit_end` closes it. If you read
   the callback in isolation you won't see where the run came from.

4. **`{run_dir}/artifacts/` ≠ `{LAKE_ROOT}/mlartifacts/`.** First is analyzer
   output (per-user, under `RUN_ROOT`). Second is MLflow's declared-but-unused
   artifact root (shared). Don't confuse.

5. **Query path is MLflow, not OTel.** `traces.jsonl` is for single-run
   debugging; don't build cross-run analysis on it. Use `mlflow.search_runs`
   + `client.get_metric_history`. All graphids-identity filter_strings flow
   through `graphids._mlflow.build_search_filter(...)` — don't hand-compose.

6. **`atomic_save`/`atomic_load` hashing is load-time integrity** on GPFS —
   independent of MLflow's provenance tagging. Both serve real purposes; don't
   collapse.

7. **Experiments are per-axis (post-2026-04-24).** Layout is
   `graphids/{dataset}/{group}`, not the old `graphids/{group}/{variant}`.
   Historical rows remain in old experiment names; all queries go through
   `build_search_filter` with tag predicates, so old + new rows are found
   uniformly.

8. **Upstream lineage flows through `LoggedModel`.** Each fit's
   `MLflowTrainingCallback.on_fit_end` registers a metadata-only
   `LoggedModel` (no artifact bytes) carrying `tags.graphids.ckpt_path`,
   `ckpt_sha256`, and mirrored identity (`dataset/group/variant/seed`),
   plus `params.graphids.run_dir`. Downstream (curriculum_vgae's VGAE
   teacher, fusion's vgae+gat teachers) resolves an upstream ckpt by
   `client.search_logged_models(experiment_ids=[...], filter_string="tags.\`graphids.group\` = '...' AND tags.\`graphids.variant\` = '...' AND tags.\`graphids.seed\` = '...'")`,
   then reads `lm.tags["graphids.ckpt_path"]`. The filesystem path is
   still the load-bearing identity; `LoggedModel` is just the index.
   No `graphids.upstream.*` run-level tags exist — that scheme was
   superseded by LoggedModel before any code shipped with it.

## Anti-patterns seen before

- Adding `mlflow.log_artifact(ckpt_path)` "because the artifact dir is empty" — no
- Adding a parallel `summary.json` under run_dir "because MLflow might not have loaded" — no
- Writing custom metrics to `metrics.jsonl` "because that's what OTel does" — deleted 2026-04-16; use `mlflow.log_metrics(..., step=epoch)` via the callback
- Re-opening the fit-phase MLflow run in `evaluate` to append test metrics — rejected; separate rows share `run_name` and are distinguished by `graphids.phase`
- Hand-composing `search_runs` `filter_string=` — use `build_search_filter(...)` so old + new layout, phase gating, and cluster tag selection stay consistent
- Reading `active_run().data.tags` immediately after `set_tags` — it's a snapshot at `start_run` time, stale for post-start writes. Re-fetch via `MlflowClient().get_run(run_id)` when inspecting just-set tags
- Treating `LAKE_ROOT` and `RUN_ROOT` as the same path — they're not, on OSC `RUN_ROOT = ${LAKE_ROOT}/dev/${USER}` but each is its own env var
- Wrapping every MLflow call in `try/except Exception` — MLflow is a hard dep, failures propagate. The only legitimate soft-failures are documented in `_mlflow.py` module docstring
