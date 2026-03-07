# KD-GAT SLURM / HPC Conventions

## Environment

- **Cluster**: OSC Pitzer (Ohio Supercomputer Center), RHEL 9, SLURM
- **Account**: PAS1266 (`$KD_GAT_SLURM_ACCOUNT` in `.env`). Must `source .env` before `sbatch` on login node.
- **GPU**: 2x V100 per node, ~362 GB RAM, gpu partition
- **Python**: 3.12 via `module load python/3.12`, uv venv `.venv/`
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)

## Rules

- Spawn/fork CUDA rule: See critical-constraints.md.
- Test on small datasets (`hcrl_ch`) before large ones (`set_02`+).
- SLURM logs go to `slurm_logs/`, experiment outputs to `experimentruns/`.
- Heavy tests use `@pytest.mark.slurm` — auto-skipped on login nodes.
- **Always run tests via SLURM** (`cpu` partition, 8 CPUs, 16GB). Submit with `bash scripts/slurm/run_tests_slurm.sh`.

## Writing SLURM Job Scripts

When creating or modifying a SLURM `.sh` script, follow these conventions:

### Resource Sizing

| Resource | Default | When to increase |
|----------|---------|------------------|
| `--mem` | `48G` | Only if `sacct` shows MaxRSS > 40G for this job type |
| `--cpus-per-task` | `4` | Multi-worker DataLoader (`num_workers > 2`) or multi-concurrent Ray trials |
| `--gres` | `gpu:1` | Never request `gpu:v100:1` — use generic `gpu:1` for scheduler flexibility |
| `--time` | `08:00:00` | Increase for Phase B full training (200 epochs) |

**Right-size resources.** Over-requesting memory/CPUs increases your scheduler footprint and slows queue priority. Check actual usage with `sacct -j <JOBID> -o MaxRSS,ReqMem`. Historical: training jobs use 8-20G RAM; tune sweeps use 5-15G.

### Available Partitions

| Partition | Use | Max time | Notes |
|-----------|-----|----------|-------|
| `gpu` | Training, sweeps, evaluation | 7 days | 2x V100 per node |
| `gpudebug` | Smoke tests (Layer 2) | 1 hour | Priority scheduling |
| `cpu` | Tests, export, preprocessing | 7 days | No GPU |
| `debug-cpu` | Quick CPU validation | 1 hour | Priority scheduling |

**There is NO `serial` partition on Pitzer.** CPU jobs use `--partition=cpu`.

### Required SBATCH Directives

**GPU jobs:**

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=kd-gat-<descriptive>
#SBATCH --output=slurm_logs/<prefix>_%j.out
#SBATCH --error=slurm_logs/<prefix>_%j.err
#SBATCH --signal=B:USR1@300
```

**CPU jobs** (tests, export, preprocessing):

```bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=kd-gat-<descriptive>
#SBATCH --output=slurm_logs/<prefix>_%j.out
#SBATCH --error=slurm_logs/<prefix>_%j.err
```

CPU preamble: `SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh"`

### Required Environment Setup

All boilerplate is in shared sourced scripts:

```bash
# Preamble: env setup, venv, .env, CUDA config, data staging
source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh"

# Override before sourcing:
#   SKIP_CUDA_CONF=1   — for CPU-only jobs
#   SKIP_STAGE_DATA=1  — skip data staging
#   STAGE_DATA_ARGS="--raw"  — for preprocessing jobs
```

```bash
# Epilog: GPU utilization report
JOB_LOG_PREFIX="ray" source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_epilog.sh"
```

### Key Patterns

- **`--signal=B:USR1@300`** — sends USR1 five minutes before wall time for graceful shutdown
- **`_preamble.sh`** — sets up Python 3.12, venv, .env, CUDA memory config, data staging
- **`_epilog.sh`** — GPU utilization report (resource right-sizing)

## Data Staging Protocol

Data staging uses a 3-tier storage hierarchy. `scripts/data/stage_data.sh` manages this automatically.

### Storage Tiers

| Tier | Path | Speed | Persistence | Use |
|------|------|-------|-------------|-----|
| **NFS** (home) | `~/KD-GAT/data/` | Slow | Permanent | Source of truth |
| **Scratch** (GPFS) | `/fs/scratch/PAS1266/kd-gat-data/` | Fast | 90-day purge | Shared across jobs |
| **TMPDIR** (local SSD) | `$TMPDIR/kd-gat-data/` | Fastest | Per-job only | Training I/O |

### Smart Caching

`stage_data.sh` uses marker files (`.staged_marker`) to skip redundant copies:

1. **NFS → Scratch**: Skipped if marker exists and source file count matches. The 90-day scratch purge deletes the marker, triggering a fresh sync automatically.
2. **Scratch → TMPDIR**: Skipped if `$TMPDIR/kd-gat-data/cache/` already exists (shouldn't happen since TMPDIR is per-job, but guards against re-sourcing).

### When to Re-stage

- After rebuilding preprocessed caches (`cache-*.sh` jobs) — new files change the count, marker auto-invalidates
- After scratch purge (90 days idle) — marker is deleted, next job auto-syncs
- Manual: `rm /fs/scratch/PAS1266/kd-gat-data/cache/.staged_marker` to force re-sync

## 3-Layer Validation Protocol

**Before submitting any new or modified job to `gpu` partition, validate in layers:**

### Layer 1: Dry-Run (login node, no GPU, ~2s)

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m graphids.pipeline.cli tune \
    --dry-run --model <stage> --dataset <dataset> --scale <scale>
```

Validates: config resolution, data directory existence, search space construction, Ray imports, Tuner construction, subprocess command. Catches ~80% of errors (config typos, missing data, import issues, wrong paths).

### Layer 2: GPU Smoke Test (gpudebug partition, ~5-10 min)

```bash
sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/smoke_test.sh <stage>
```

Runs 1 trial, 2 epochs, on `hcrl_ch` (smallest dataset). Uses `gpudebug` partition (1hr max, priority scheduling — starts within minutes). Validates: GPU access, CUDA context, data loading, model forward/backward pass, checkpoint save, metrics reporting.

### Layer 3: Production (gpu partition)

Only after Layer 2 passes. Submit the real job with full dataset/samples.

### When to Use Each Layer

| Change | Layer 1 | Layer 2 | Layer 3 |
|--------|---------|---------|---------|
| New config field / YAML change | Required | Required | Then submit |
| Code change in model/training | Required | Required | Then submit |
| New SLURM script or env change | Skip | Required | Then submit |
| Same code, different dataset | Required | Skip | Submit |
| Re-run after OOM/timeout | Skip | Skip | Adjust resources, submit |

## Login Node Safety

**Safe on login node:**
- Import checks: `python -c "from graphids.config import resolve; print('OK')"`
- DuckDB queries: `duckdb < data/datalake/queries/leaderboard.sql`
- Git, DVC, ruff

**Must go through SLURM:**
- `python -m graphids.pipeline.cli <any stage>` — all training/evaluation
- `python -m pytest` — test suite
- Any script that imports and runs models
