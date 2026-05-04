# GraphIDS Data Layout — Store Ownership

**Three roots — keep them straight.** `LAKE_ROOT` (shared, cross-user)
holds metadata + caches everyone reads. `RUN_ROOT` (per-user) holds this
user's experiment writes. `run_dir` is the per-(variant,seed) leaf under
`RUN_ROOT`. Conflating them produced the 2026-04-24 jsonnet/Python
default drift — impossible by construction now.

> Filesystem tree + per-file write-path inventory:
> `docs/reference/write-paths.md`. This file owns the **store-ownership
> table** and the rules that prevent duplicating stores.

## Cache partitioning

Caches under `{LAKE_ROOT}/cache/v{PREPROCESSING_VERSION}/{dataset}/voc_{scope}/`.
The `voc_{scope}` partition (added 2026-05-04) lets `vocab_scope="train"`
and `vocab_scope="all"` regimes coexist on disk for ablation. See
`BaseGraphSource.build()` and `core/data/datasets/_base.py`.

## Store ownership — don't duplicate

| Signal | Where | Reader |
|---|---|---|
| Run metadata (params, tags, timestamps) | `mlflow.db` | `mlflow.search_runs` + `_mlflow.build_search_filter(...)` |
| Per-epoch scalar metrics (train/val_loss, lr) | `mlflow.db` | `client.get_metric_history(run_id, key)` |
| Device telemetry (GPU util, VRAM, CPU, mem) | `mlflow.db` (system-metrics sampler) | MLflow UI charts |
| Final test metrics (incl. `test/{subdir}/auroc`, `auroc_per_attack/{name}`) | `mlflow.db` | `search_runs` filter `tags.graphids.phase = 'test'` |
| Dataset identity (cache digest) | `mlflow.db` `Dataset` entity | `search_runs` filter `dataset.digest = '...'` |
| LoggedModel (ckpt metadata, upstream lineage) | `mlflow.db` | `search_logged_models(...)` |
| Checkpoint bytes | `{run_dir}/checkpoints/` | `torch.load` via `_fs.atomic_load` |
| Checkpoint SHA256 | `.sha256` sidecar + `LoggedModel.tags.graphids.ckpt_sha256` | `atomic_load` verifies |
| Span lifecycle + log events | `{run_dir}/traces.jsonl` | `jq` for debugging |
| Validated jsonnet config | `{run_dir}/resolved.json` | replay |

## Key rules

1. **Never `mlflow.log_artifact` or `mlflow.pytorch.log_model`.** Checkpoints
   stay as filesystem paths so resume / KD-student teacher loading / fusion
   upstream-ckpt all work via direct paths. MLflow rows link via the
   `graphids.run_dir` tag. MLflow-3 `LoggedModel` (metadata-only via
   `MlflowClient.create_logged_model`) is OK — no artifact bytes.

2. **Two MLflow rows per run, same `run_name`.** Fit phase (`graphids.phase='fit'`)
   and test phase (`='test'`). `run_name` is
   `{group}_{variant}_{dataset}_seed{N}[_{cluster}]`. Filter by phase tag.
   Fit resumable (FAILED/KILLED → same `run_id`); test always-fresh.

3. **`{run_dir}/artifacts/` ≠ `{LAKE_ROOT}/mlartifacts/`.** First is analyzer
   output (per-user). Second is MLflow's declared-but-unused artifact root.

4. **Query path is MLflow.** `traces.jsonl` is single-run debug only. All
   graphids-identity filters flow through `_mlflow.build_search_filter(...)`.

5. **Experiments are per-axis** (post-2026-04-24): `graphids/{dataset}/{group}`.
   Queries via `build_search_filter` find old + new layouts uniformly.

6. **Upstream lineage flows through `LoggedModel`.** Each fit's callback
   registers a metadata-only LM carrying `tags.graphids.ckpt_path`,
   `ckpt_sha256`, mirrored identity. Downstream resolves via
   `client.search_logged_models(...)` then reads `lm.tags["graphids.ckpt_path"]`.
   Filesystem path is still the load-bearing identity; LM is the index.

## Anti-patterns

- `mlflow.log_artifact(ckpt_path)` or parallel `summary.json` under run_dir — no, paths are the source of truth.
- Custom `metrics.jsonl` — deleted 2026-04-16; use `mlflow.log_metrics(..., step=epoch)`.
- Re-opening fit-phase run in `evaluate` to append test metrics — separate rows, distinguished by `graphids.phase`.
- Hand-composing `search_runs` `filter_string=` — use `build_search_filter(...)`.
- Reading `active_run().data.tags` immediately after `set_tags` — stale snapshot; re-fetch via `MlflowClient().get_run(run_id)`.
- Wrapping every MLflow call in `try/except` — MLflow is a hard dep, exceptions propagate.
