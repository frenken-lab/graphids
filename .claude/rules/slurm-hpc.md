# GraphIDS SLURM / HPC Conventions

## Environment

- **Cluster**: OSC Pitzer (Ohio Supercomputer Center), RHEL 9, SLURM
- **Account**: PAS1266 (`$GRAPHIDS_SLURM_ACCOUNT` in `.env`). Must `source .env` before `sbatch` on login node.
- **GPU**: 2x V100 per node, ~362 GB RAM, gpu partition
- **Python**: 3.12 via `module load python/3.12`, uv venv `.venv/`
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)

## Rules

- Spawn/fork CUDA rule: See critical-constraints.md.
- Test on small datasets (`hcrl_ch`) before large ones (`set_02`+).
- SLURM logs go to `slurm_logs/`, experiment outputs to `experimentruns/`.
- Heavy tests use `@pytest.mark.slurm` — auto-skipped on login nodes.
- **Always run tests via SLURM.** Submit with `scripts/slurm/submit.sh tests [-k pattern]`.

## Job Submission

All SLURM jobs are submitted via the unified launcher `scripts/slurm/submit.sh <job> [args...]`:

| Job | Command |
|-----|---------|
| Tests | `scripts/slurm/submit.sh tests [-k pattern] [-x]` |
| Cache rebuild | `scripts/slurm/submit.sh rebuild-caches --all --delete-existing --yes` |
| Analyze ckpt  | `scripts/slurm/submit.sh analyze --ckpt-path <p> --dataset <name>` |
| Pipeline run | `scripts/slurm/submit.sh pipeline-run --dataset hcrl_sa` |
| Extract fusion states | `scripts/slurm/submit.sh extract-fusion-states` |
| Profiling | `scripts/slurm/submit.sh profile` |

Source of truth for the table: `configs/resources/submit_profiles.json` —
`python -m graphids submit-profile <job>` prints the resource tuple
that `submit.sh` consumes.

submit.sh handles `.env` sourcing, account selection, and resource defaults.

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

CPU preamble: `SKIP_CUDA_CONF=1 source "$SCRIPT_DIR/_preamble.sh"`

### Required Environment Setup

All boilerplate is in shared sourced scripts:

```bash
# Auto-detect project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Preamble: env setup, venv, .env, CUDA alloc config
source "$SCRIPT_DIR/_preamble.sh"

# Override before sourcing:
#   SKIP_CUDA_CONF=1   — for CPU-only jobs
```

```bash
# Epilog: GPU utilization report
JOB_LOG_PREFIX="ray" source "$SCRIPT_DIR/_epilog.sh"
```

### Key Patterns

- **`--signal=B:USR1@300`** — sends USR1 five minutes before wall time for graceful shutdown
- **`_preamble.sh`** — sets up Python 3.12, venv, .env, CUDA memory config
- **`_epilog.sh`** — GPU utilization report (resource right-sizing)

## Data I/O

Jobs read raw CSVs and cache tensors directly from ESS NFS
(`/fs/ess/PAS1266/graphids/{raw,cache}/`). The old `stage-data` command
(NFS → scratch → TMPDIR) was removed 2026-04-14 after rebuild confirmed
direct NFS reads are fast enough for our working set. If training ever
becomes I/O-bound, reintroduce a real staging command — don't paper over
with a silent eval.

## Login Node Safety

**Safe on login node:**
- Import checks: `python -c "from graphids.config import schemas; print('OK')"`
- DuckDB queries: `duckdb < data/datalake/queries/leaderboard.sql`
- Git, ruff

**Must go through SLURM:**
- `python -m graphids fit|test|validate|predict` — all training/evaluation
- `python -m pytest` — test suite
- Any script that imports and runs models
