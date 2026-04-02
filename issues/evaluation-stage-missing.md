# No evaluation (test set) stage in pipeline — RESOLVED

> **Status**: Fixed in session 5 (2026-04-01). `test-from-spec` command added. Every training SLURM job now runs train→test→analyze sequentially. No separate dagster eval asset needed.

## Problem

The pipeline DAG has training stages (autoencoder, curriculum/normal, fusion) but no evaluation stage that runs `LightningCLI test` on the held-out test set. After training completes, there's no automated way to get test metrics.

Currently the only way to evaluate is manual:
```bash
python -m graphids test --config <stage>.yaml --ckpt_path <path>
```

## What's needed

An `evaluation` asset downstream of each training asset that:
1. Takes the best checkpoint path as input (from IOManager)
2. Submits `python -m graphids test --config <stage>.yaml --ckpt_path <ckpt>` via SLURM
3. Outputs test metrics (accuracy, F1, AUC per attack type) as asset metadata
4. Writes metrics to the DuckDB catalog for leaderboard queries

## DAG shape

```
autoencoder ──→ autoencoder_eval
curriculum  ──→ curriculum_eval
fusion      ──→ fusion_eval
```

Each eval asset depends only on its training asset. Eval jobs are short (single forward pass over test set) — `gpudebug` partition, ~5 min wall time.

## Recipe integration

Recipes should have an `evaluate: true/false` toggle (default true). Smoke tests may want to skip eval to save queue time.
