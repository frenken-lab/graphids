---
name: sync-state
description: Update STATE.md from current pipeline outputs and display it
---

Manually regenerate STATE.md from current experiment outputs.

## Execution Steps

1. **Scan experiment runs** to gather current status:
   ```bash
   # List all completed stages with timestamps
   for ds in hcrl_ch hcrl_sa set_01 set_02 set_03 set_04; do
     echo "=== $ds ==="
     ls -lh experimentruns/$ds/*/best_model.pt 2>/dev/null
     ls -lh experimentruns/$ds/*/metrics.json 2>/dev/null
   done
   ```

2. **Check MLflow status**:
   ```bash
   python -c "
   import mlflow
   from graphids.config import MLFLOW_TRACKING_URI
   mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
   runs = mlflow.search_runs(search_all_experiments=True)
   print(f'MLflow runs: {len(runs)}')
   if not runs.empty:
       print(runs.groupby('status').size())
   " 2>/dev/null || echo "No MLflow data yet"
   ```

3. **Update `.claude/system/STATE.md`** with current findings:
   - What's working (config, pipeline, dashboard)
   - Current experiment status per dataset
   - What's incomplete or broken
   - Next steps

4. **Read and display the updated STATE.md** to the user.

## Notes

- STATE.md is the primary context file for session awareness.
- Run this at the start of each session or after significant pipeline activity.
- Scans `experimentruns/` filesystem and MLflow for current state.
