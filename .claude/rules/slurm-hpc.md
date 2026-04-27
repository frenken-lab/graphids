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
- **Always run tests via SLURM.** Submit with `python -m graphids submit --mode cpu --length short --command "python -m pytest [-k pattern]"`.

## Job Submission

One Typer command, `python -m graphids submit`, two shapes:

**Training / ablations: `python -m graphids submit <preset.jsonnet> [options]`.**
The preset owns run specifics; flags map to TLAs internally so you never
type nested JSON quotes. Defaults to `gpu` mode + `long` length (per-cluster
wall in `submit_profiles.json`); `--smoke` swaps to `short` (gpudebug 1hr).
`--depends-on <variant>[:<seed>]` is the **only** dep mechanism: FINISHED
upstream → inject ckpt TLA; RUNNING upstream → also add its
`slurm.slurm_job_id` as an `afterok` dep. One primitive — no separate
`--dep` flag, no `SBATCH_DEP` env fallback. See
`.claude/rules/single-submission-primitive.md`. Full flag list via
`python -m graphids submit --help`. Backed by `submitit.AutoExecutor`;
library entrypoint is `graphids.slurm.submit.submit()`.

```bash
python -m graphids submit configs/ablations/unsupervised/vgae.jsonnet --dataset set_01 --seed 42
python -m graphids submit configs/ablations/fusion/dqn.jsonnet \
    --dataset set_01 --seed 42 --depends-on vgae:42,focal:42 --cluster cardinal
```

**Ops: `python -m graphids submit --mode {gpu|cpu} --command "..." [--mem-gb N --timeout-min M --length short|long]`.**

No per-job profile registration.

| Job | Command |
|-----|---------|
| Tests | `python -m graphids submit --mode cpu --length short --command "python -m pytest [-k pattern]"` |
| Cache rebuild | `python -m graphids submit --mode cpu --mem-gb 54 --timeout-min 240 --command "python -m graphids rebuild-caches --all --delete-existing --yes"` |
| Analyze ckpt | `python -m graphids submit --mode gpu --mem-gb 32 --timeout-min 120 --command "python -m graphids analyze --ckpt-path <p> --dataset <name>"` |
| Extract fusion states | `python -m graphids submit --mode gpu --mem-gb 36 --timeout-min 30 --command "python -m graphids extract-fusion-states ..."` |
| Profiling | `python -m graphids submit --mode gpu --length short --command "python -m graphids profile"` |

Source of truth for `partition` + `cpus_per_task` + `mem_gb` + `timeout_min`:
`configs/resources/submit_profiles.json`, keyed `[mode][cluster][length]`.
Entries are raw submitit AutoExecutor kwargs — no translation layer.
Per-job overrides flow through flags (`--mem-gb`, `--timeout-min`). Optional
`--time-from-history` opts into MLflow-history walltime estimation
(library: `graphids.slurm.sizing`).

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

- **Preemption auto-resume** — profiles set `slurm_signal_delay_s=300`, which makes submitit's sbatch emit `--signal=USR2@300`. SIGUSR2 five minutes before walltime triggers `_TrainingJob.checkpoint()` → `DelayedSubmission` with `ckpt_path={run_dir}/checkpoints/last.ckpt` → submitit sbatch-queues the resumed job via afterany. No manual resubmit loop. (USR2 because NCCL catches USR1.)
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
- `python -m graphids fit|test` — all training/evaluation
- `python -m pytest` — test suite
- Any script that imports and runs models
