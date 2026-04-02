# Consolidate SLURM Jobs: train → test → analyze in one job

> **Status: IMPLEMENTED** (2026-04-01). Supersedes `evaluation-analysis-assets.md` (deleted).
> Implementation details in git history.

## Design

Each model config = one dagster asset = one SLURM job running 3 sequential commands:

```bash
#!/bin/bash
source scripts/slurm/_preamble.sh
python -m graphids train-from-spec --spec-file $SPEC_FILE
python -m graphids test-from-spec  --spec-file $SPEC_FILE
python -m graphids analyze-from-spec --spec-file $SPEC_FILE
source scripts/slurm/_epilog.sh
```

`set -euo pipefail` in preamble: if train fails, test and analyze don't run.
Test and analyze are best-effort (`set +euo pipefail` before those phases).
Per-phase markers: `.train_complete`, `.test_complete`, `.analyze_complete`.

## Why (bugs this fixed)

| Bug | How consolidation fixed it |
|-----|---------------------------|
| Analysis ran in-process on CPU dagster worker | Analyze runs inside GPU SLURM job |
| No evaluation stage existed | `test-from-spec` runs `LightningCLI test` in same job |
| Multiprocess executor child failures | Fewer assets = fewer executor workers |
| Recipe env var lost across processes | Fewer process boundaries |

## Key files

| File | Role |
|------|------|
| `graphids/commands/test_from_spec.py` | test-from-spec command |
| `graphids/core/train_entrypoint.py` | `run_test_from_spec()` |
| `graphids/slurm/slurm.py` | `generate_script()` — multi-command |
| `graphids/orchestrate/assets.py` | Single asset per model config |
| `graphids/orchestrate/checks.py` | Unified checkpoint + phase checks |

## What did NOT change

TrainingSpec, AnalysisSpec, ConfigResolver, enumerate_assets/StageConfig,
_preamble.sh/_epilog.sh, IOManager checkpoint handoff.
