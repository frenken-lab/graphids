---
name: check-status
description: Check status of running experiments, SLURM jobs, and pipeline progress
---

Check the status of experiments and running jobs.

## Arguments

`$ARGUMENTS` - Optional dataset name to filter (e.g., `hcrl_sa`). If empty, check all datasets.

## Execution Steps

1. **Check SLURM job queue**
   ```bash
   squeue -u $USER --format="%.10i %.20j %.8T %.10M %.6D %.15R" 2>&1
   ```

2. **Check experiment checkpoints** for all datasets (or filtered dataset)
   ```bash
   # List all completed stages
   for ds in hcrl_ch hcrl_sa set_01 set_02 set_03 set_04; do
     echo "=== $ds ==="
     ls -lh experimentruns/$ds/*/best_model.pt 2>/dev/null || echo "  (no checkpoints)"
   done
   ```

3. **Check for recent SLURM errors** in slurm_logs directory
   ```bash
   ls -lt slurm_logs/*.err 2>/dev/null | head -10
   ```

4. **Check MLflow for recent runs**
   ```bash
   python -c "
   import mlflow
   from graphids.config import MLFLOW_TRACKING_URI
   mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
   runs = mlflow.search_runs(max_results=5, order_by=['start_time DESC'])
   if runs.empty:
       print('  (no MLflow runs)')
   else:
       print(runs[['run_id','tags.mlflow.runName','status','start_time']].to_string())
   " 2>/dev/null || echo "  (MLflow not available)"
   ```

5. **If dataset specified via `$ARGUMENTS`**, show detailed status:
   ```bash
   ls -la experimentruns/$ARGUMENTS/*/best_model.pt 2>/dev/null
   ls -la experimentruns/$ARGUMENTS/*/config.json 2>/dev/null
   ls -la experimentruns/$ARGUMENTS/*/metrics.json 2>/dev/null
   ```

## Output Summary

Provide a concise status report:

| Dataset | Stage | Status | Last Updated |
|---------|-------|--------|--------------|
| hcrl_sa | vgae_large_autoencoder | complete/missing | timestamp |
| hcrl_sa | gat_large_curriculum | complete/missing | timestamp |
| hcrl_sa | dqn_large_fusion | complete/missing | timestamp |
| hcrl_sa | vgae_small_autoencoder_kd | complete/missing | timestamp |
| ... | ... | ... | ... |

## Useful Follow-up Commands

```bash
# Watch job queue
watch -n 5 'squeue -u $USER'

# Follow specific SLURM log
tail -f slurm_logs/<jobid>.err

# Browse MLflow runs
mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db
```
