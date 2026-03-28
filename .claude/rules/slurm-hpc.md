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

## Shell Building Blocks (`scripts/lib/`)

**When writing new shell scripts, compose from `scripts/lib/` functions instead of writing ad-hoc code.** Source the modules you need:

```bash
source "$(dirname "$0")/../lib/datasets.sh"   # KD_ALL_DATASETS, kd_parse_datasets, kd_each_dataset
source "$(dirname "$0")/../lib/dryrun.sh"      # kd_parse_dry_run, kd_exec, kd_mkdir
source "$(dirname "$0")/../lib/slurm.sh"       # kd_submit, kd_sbatch_gpu_args, kd_sbatch_cpu_args
source "$(dirname "$0")/../lib/validation.sh"  # kd_run_dir, kd_check_checkpoint, kd_run_complete
# _bootstrap.sh (kd_log, kd_die, kd_load_env) is auto-sourced by all above
```

**Conventions:** `kd_` prefix on all functions. Source guards prevent double-loading. Functions return exit codes, never call `exit` (except `kd_die`). Works alongside `_preamble.sh` / `_epilog.sh`.

**Example (login-node launcher):**
```bash
#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../lib/datasets.sh"
source "$(dirname "$0")/../lib/dryrun.sh"
source "$(dirname "$0")/../lib/slurm.sh"

kd_parse_dry_run "$@"
read -ra DATASETS <<< "$(kd_parse_datasets "$@")"

do_one() { kd_submit gpu "train-$1" "source .../scripts/slurm/_preamble.sh && python -m graphids dataset=$1"; }
kd_each_dataset do_one "${DATASETS[@]}"
```

## Writing SLURM Job Scripts

When creating or modifying a SLURM `.sh` or '.sbatch' script, follow these conventions:

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
```

**CPU jobs** (tests, export, preprocessing):

```bash
#SBATCH --partition=cpu
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

## Login Node Safety

**Safe on login node:**
- Import checks: `python -c "from graphids.config import resolve; print('OK')"`
- DuckDB queries: `duckdb < data/datalake/queries/leaderboard.sql`
- Git, ruff

**Must go through SLURM:**
- `python -m graphids.cli <any stage>` — all training/evaluation
- `python -m pytest` — test suite
- Any script that imports and runs models
