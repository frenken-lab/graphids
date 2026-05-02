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
- **Never run `pytest` on login nodes.** Submit via `python -m graphids submit --mode cpu --length short --command "python -m pytest [-k pattern]"`.

## Job Submission

One Typer command, `python -m graphids submit`, two shapes:

**Training / ablations: `python -m graphids submit <preset.jsonnet> [options]`.**
The preset owns run specifics; flags map to TLAs internally so you never
type nested JSON quotes. Defaults to `gpu` mode + `long` length (per-cluster
wall in `submit_profiles.json`); `--smoke` swaps to `short` (gpudebug 1hr).
`--depends-on <variant>[:<seed>]` is the **only** dep mechanism: FINISHED
upstream → inject ckpt TLA; RUNNING upstream → also add its
`slurm.job_id` as an `afterok` dep. One primitive — no separate
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

**Preemption auto-resume** — profiles set `slurm_signal_delay_s=300`, so
submitit's sbatch emits `--signal=USR2@300`. SIGUSR2 five minutes before
walltime triggers `_TrainingJob.checkpoint()` → `DelayedSubmission` with
`ckpt_path={run_dir}/checkpoints/last.ckpt` → submitit sbatch-queues the
resumed job via afterany. No manual resubmit loop. (USR2 because NCCL
catches USR1.) `scripts/slurm/_preamble.sh` is sourced inside the sbatch
shell via `slurm_setup`; `_epilog.sh` reports GPU utilization.

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
