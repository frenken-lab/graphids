# Analysis assets run in-process (no SLURM submission) — RESOLVED

> **Status**: Fixed in session 5 (2026-04-01). Analysis now runs inside GPU SLURM job via `analyze-from-spec`. `make_analysis_asset()` deleted; `make_training_asset()` passes `AnalysisSpec` to `submit_and_wait()`.

## Problem

`make_analysis_asset()` in `orchestrate/assets.py:131` calls `run_analysis_from_spec(spec)` directly in the dagster worker process. This:

1. Imports torch at definition time (`assets.py:9`) — violates "dagster must never import torch" rule
2. Runs model inference on the CPU orchestrator node — no GPU available
3. NVML warnings already appearing in orchestrator logs (job 46256235)

Training assets correctly use `context.resources.slurm.submit_and_wait()` to submit GPU work via sbatch. Analysis assets need the same pattern.

## Fix

- Add `analyze-from-spec` as a SLURM-submitted job (same as `train-from-spec`)
- `make_analysis_asset()` should call `slurm.submit_and_wait()` with an `AnalysisSpec`
- Remove `from graphids.core.analyze_entrypoint import run_analysis_from_spec` from `assets.py` top-level
- Analysis jobs use GPU or CPU partition depending on model type (VGAE landscape needs GPU, embeddings extraction is CPU-ok)

## Evidence

- `assets.py:9`: `from graphids.core.analyze_entrypoint import run_analysis_from_spec`
- `assets.py:131`: `run_analysis_from_spec(spec)` — in-process call
- `slurm_logs/ablation_46256266.err`: 22x "Can't initialize NVML" from torch import on CPU node
- Training assets (`assets.py:77`): correctly use `context.resources.slurm.submit_and_wait()`
