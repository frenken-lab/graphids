# GraphIDS Data Layout — Where Things Go and Why

Two write roots: the **lake** (cross-run catalog) and the **run_dir** (per-run
filesystem tree). Knowing which is which and what lives in each is the main
thing to get right before touching anything in `graphids/_mlflow.py`,
`graphids/orchestrate/stage.py`, or `graphids/core/callbacks.py`.

## The tree

```
{LAKE_ROOT}/                                  # /fs/ess/PAS1266/graphids/dev/rf15 in prod
├── mlflow.db                                 # MLflow SQLite backend — SOURCE OF TRUTH for:
│                                             #   run identity + timestamps, params (flattened
│                                             #   resolved.json), per-epoch scalar metrics
│                                             #   (via get_metric_history), final metrics,
│                                             #   tags (graphids.*, git_sha, slurm.*)
├── mlartifacts/                              # MLflow artifact store — INTENTIONALLY EMPTY.
│                                             #   MLflow requires artifact_location per
│                                             #   experiment but graphids keeps ckpts on the
│                                             #   filesystem (see below). Do NOT call
│                                             #   mlflow.log_artifact / mlflow.pytorch.log_model.
│
└── {dataset}/ablations/{group}/{variant}/seed_{N}/    # run_dir (jsonnet _paths.libsonnet)
    ├── checkpoints/
    │   ├── best_model.ckpt                   # ModelCheckpoint via _fs.atomic_save
    │   ├── best_model.ckpt.sha256            # integrity sidecar — VERIFIED on load by
    │   │                                     #   atomic_load; also stamped as MLflow tag
    │   ├── last.ckpt
    │   └── last.ckpt.sha256
    ├── predictions/
    │   ├── train.pt / val.pt                 # stage.train post-fit predict_step
    │   └── test/{set_name}.pt                # stage.evaluate per-test-subdir
    ├── artifacts/                            # `graphids analyze` output (UMAP, CKA, confusion
    │                                         #   matrices). DIFFERENT FROM {lake}/mlartifacts/.
    ├── traces.jsonl                          # OTel: single training.fit span + structured log
    │                                         #   events (budget_probed, vram_drift_detected).
    │                                         #   Not a cross-run query surface.
    ├── resolved.json                         # Pydantic-validated jsonnet render
    ├── overrides.json                        # TLA dict + --set payload
    ├── .train_complete / .test_complete      # phase markers (diagnostic only; resume reads
    │                                         #   checkpoints/best_model.ckpt directly)
```

## Store ownership — don't duplicate

| Signal | Where | Reader |
|---|---|---|
| Run metadata (params, tags, timestamps) | `mlflow.db` | `mlflow.search_runs` |
| Per-epoch scalar metrics (train/val_loss, lr) | `mlflow.db` | `client.get_metric_history(run_id, key)` |
| Device telemetry (GPU util, VRAM, CPU, mem) | `mlflow.db` (system-metrics sampler, 5s) | MLflow UI charts |
| Final test metrics | `mlflow.db` (test-phase row) | `search_runs` filter `tags.graphids.phase = 'test'` |
| Checkpoint bytes | `{run_dir}/checkpoints/` | `torch.load` via `_fs.atomic_load` |
| Checkpoint SHA256 | `.sha256` sidecar + MLflow tag `graphids.ckpt_sha256` | `atomic_load` verifies; tag is provenance |
| Span lifecycle + log events | `{run_dir}/traces.jsonl` | `jq` / manual grep for debugging |
| Validated jsonnet config | `{run_dir}/resolved.json` | replay / debugging |

## Key rules

1. **Never `mlflow.log_artifact` or `mlflow.pytorch.log_model`.** Checkpoints
   stay as filesystem paths so resume, KD-student teacher loading, and fusion
   upstream-ckpt flow all work via direct paths. MLflow rows link to them via
   the `graphids.run_dir` tag.

2. **Two MLflow rows per run under the same `run_name`.** Fit phase (`tags.graphids.phase = 'fit'`)
   and test phase (`tags.graphids.phase = 'test'`). `run_name` is deterministic:
   `{group}_{variant}_{dataset}_seed{N}[_{cluster}]`. Filter by the phase tag
   when `search_runs` returns what looks like duplicates.

3. **MLflow run lifecycle spans `stage.train` + the callback.**
   `_mlflow.start_training_run` opens the run in `stage.train::train` before
   `trainer.fit`; `MLflowTrainingCallback.on_fit_end` closes it. If you read
   the callback in isolation you won't see where the run came from.

4. **`{run_dir}/artifacts/` ≠ `{lake_root}/mlartifacts/`.** First is analyzer
   output. Second is MLflow's declared-but-unused artifact root. Don't confuse.

5. **Query path is MLflow, not OTel.** `traces.jsonl` is for single-run
   debugging; don't build cross-run analysis on it. Use `mlflow.search_runs`
   + `client.get_metric_history`.

6. **`atomic_save`/`atomic_load` hashing is load-time integrity** on GPFS —
   independent of MLflow's provenance tagging. Both serve real purposes; don't
   collapse.

## Anti-patterns seen before

- Adding `mlflow.log_artifact(ckpt_path)` "because the artifact dir is empty" — no
- Adding a parallel `summary.json` under run_dir "because MLflow might not have loaded" — no
- Writing custom metrics to `metrics.jsonl` "because that's what OTel does" — deleted 2026-04-16; use `mlflow.log_metrics(..., step=epoch)` via the callback
- Re-opening the fit-phase MLflow run in `evaluate` to append test metrics — rejected; separate rows share `run_name` and are distinguished by `graphids.phase`
