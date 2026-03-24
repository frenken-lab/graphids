# Ablation Run 001 — Status

> Submitted: 2026-03-24 00:17 | Datasets: set_01, set_02 | Seed: 42 | Scale: small

## Job Summary

| State | Count | Notes |
|-------|-------|-------|
| COMPLETED | 3 | GPS autoencoder (batch=512) set_01+set_02, GPS autoencoder (batch=1024) set_01 |
| RUNNING | 2 | 45972988, 45972990 — normal stage (set_01 focal, set_02 weighted_ce) |
| TIMEOUT | 4 | 2 GPU training (wall=120min), 2 CPU eval (wall=30min, stuck on data staging) |
| FAILED | 13 | 7 GPU timeout-as-fail (submitit UncompletedJobError), 3 GPS OOM, 3 GPS dep cascade |
| CANCELLED | 1 | GPS dep from cancelled resubmit |
| PENDING (dep) | 46 | Downstream fusion/eval jobs waiting on upstream |

## Completed Jobs

| JobID | Config | Dataset | Elapsed | MaxRSS | Peak VRAM |
|-------|--------|---------|---------|--------|-----------|
| 45973218 | conv_gps autoencoder | set_01 | 7m | 5.7G | ~3.9G |
| 45973219 | conv_gps autoencoder | set_02 | 45m | 14.3G | ~14.8G |
| 45973072 | conv_gps autoencoder (batch=1024) | set_01 | 27m | 12.1G | — |

## Failed: GPU Timeout (~2hr wall, needed ~2.5hr)

| JobID | Config | Dataset | MaxRSS | Peak VRAM | Issue |
|-------|--------|---------|--------|-----------|-------|
| 45972985 | vgae autoencoder | set_02 | 15.1G | ~5.7G | wall=120min, needed ~150min |
| 45972987 | gat normal | set_02 | 12.0G | ~5.7G | wall=120min |
| 45972989 | gat normal | set_02 | 6.7G | ~4.5G | wall=120min |
| 45972991 | gat normal | set_02 | 12.0G | ~4.5G | wall=120min |
| 45972992 | conv_gatv1 autoencoder | set_01 | 5.3G | ~5.6G | wall=120min |
| 45972993 | conv_gatv1 autoencoder | set_02 | 15.0G | ~5.6G | wall=120min |
| 45972996 | unsup_gae autoencoder | set_01 | 5.2G | ~5.7G | wall=120min |

## Failed: GPS OOM

| JobID | batch_size | Budget (nodes) | Attempted alloc | Dataset |
|-------|-----------|----------------|-----------------|---------|
| 45972994 | 4096 | 167,936 | 105 GB | set_01 |
| 45972995 | 4096 | 212,992 | 169 GB | set_02 |
| 45973073 | 1024 | ~53,248 | 10.5 GB | set_02 |

## Failed: CPU Timeout (data staging)

| JobID | Wall | CPU time | Issue |
|-------|------|----------|-------|
| 45972998 | 30m | 1m 9s | Single-core `cp -r`, never started eval |
| 45972999 | 30m | 1m 10s | Single-core `cp -r`, never started eval |

## Efficiency Analysis

### GPU VRAM utilization (from DeviceStatsMonitor)
- Non-GPS models: **4-6 GB peak** of 16 GB V100 = **33-42% utilization**
- GPS models: **14.8 GB peak** = 94% utilization (correctly sized)
- Models are tiny (100K params) — VRAM underutilization is a batch sizing issue

### CPU RAM (MaxRSS)
- set_01 jobs: 5-6 GB (16G requested = 33% eff)
- set_02 jobs: 12-15 GB (16G requested = 75-94% eff, set_02 has larger graphs)

### Data staging
- 15-30 min per job for `cp -r` from scratch→TMPDIR (single-threaded)
- Each of 62 jobs stages independently — no write-once-read-many

## Issues to Fix Before Resubmit

1. **Wall time**: Bump training from 120→240 min (already done in resources.yaml for eval/fusion→60min)
2. **GPS batch sizing**: DynamicBatchSampler needs attention-aware budget (O(N²) vs O(N))
3. **Data staging**: Multi-process copy + shared staging across DAG jobs
4. **VRAM underutilization**: batch_size=4096 may be too small for V100 with 100K-param models

## CLI Reference

```bash
# Check status
sacct -u $USER --starttime=2026-03-24T00:17 --format="JobID%12,State%12,Elapsed%10,MaxRSS%12" --noheader | grep -v "\.batch\|\.extern"

# Check queue
squeue -u $USER

# Check specific job log
tail -20 slurm_logs/<JOBID>/<JOBID>_0_log.err
tail -20 slurm_logs/<JOBID>/<JOBID>_0_log.out
```
