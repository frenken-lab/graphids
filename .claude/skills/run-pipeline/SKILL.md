---
name: run-pipeline
description: Run Ray pipeline flow for a dataset and model configuration
---

Run the KD-GAT Ray pipeline.

## Arguments

`$ARGUMENTS` should contain: `<dataset> [scale]`

- **dataset** (required): `hcrl_sa`, `hcrl_ch`, `set_01`, `set_02`, `set_03`, `set_04`
- **scale** (optional): `large`, `small_kd`, `small_nokd`

Parse the dataset and scale from `$ARGUMENTS`. If only one word is provided, it is the dataset and all scales run.

## Usage Examples

```
/run-pipeline hcrl_sa large          # Run large pipeline for hcrl_sa
/run-pipeline hcrl_ch                # Run all scales for hcrl_ch
/run-pipeline set_01 small_kd        # Run small with KD for set_01
```

## Execution Steps

1. **Parse arguments** from `$ARGUMENTS` into dataset and optional scale.

2. **Verify dataset exists**
   ```bash
   ls data/automotive/<dataset>/
   ```

3. **Submit Ray pipeline** with appropriate arguments:
   ```bash
   PYTHONPATH=. python -m graphids.pipeline.cli flow --dataset <dataset> [--scale <scale>]
   ```
   If running on login node, add `--local` for Ray local mode (no GPU).

4. **Report the status** and show how to monitor with `squeue -u $USER`.

## Common Scales

| Scale | Description | Dependencies |
|-------|-------------|--------------|
| `large` | Full large pipeline (VGAE → GAT → DQN) | None |
| `small_kd` | Small with KD (needs large teacher) | large must complete first |
| `small_nokd` | Small without KD | None |
| (no scale) | All three scales | Runs large first, then small variants |

## Notes

- Pipeline runs on SLURM with GPU resources (V100) via Ray remote tasks
- MLflow tracking is automatic (SQLite backend, auto-pushed to HF by SLURM epilog)
- Each stage runs as a subprocess for clean CUDA context
