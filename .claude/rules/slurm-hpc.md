# GraphIDS SLURM / HPC Conventions

## Environment

- **Cluster**: OSC Pitzer / Cardinal / Ascend (RHEL 9, SLURM)
- **Account**: PAS1266 (`$GRAPHIDS_SLURM_ACCOUNT` in `.env`). `source .env` before submitting on a login node.
- **GPU**: Pitzer 2x V100 / Cardinal H100 / Ascend A100. See `~/.claude/projects/-users-PAS2022-rf15-graphids/memory/reference_osc_gpu_clusters.md`.
- **Python**: 3.12 via `module load python/3.12`, uv venv `.venv/`
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)

## Rules

- Spawn/fork CUDA rule: see `critical-constraints.md`.
- Test on small datasets (`hcrl_sa`) before large ones (`set_02`+).
- SLURM logs go to `slurm_logs/`; experiment outputs to `{RUN_ROOT}/...`.
- Heavy tests use `@pytest.mark.slurm` — auto-skipped on login nodes.
- **Never run `pytest` on login nodes.** Submit a one-row ops job.

## Job submission — four-step chassis

CLI surface is `graphids run | exec | submit`. No `python -m graphids
fit/test`, no submitit, no `--mode` ops shortcut — every job is a row.
See `single-submission-primitive.md` for the canonical example + decision rule.

The sbatch script body is a literal bash string:
`python -m graphids exec --row '<json>' [--ckpt-path X]`. **No pickle**
of Python closures — code fixes committed after submission DO reach a
pending job, because the job re-imports current source at exec time.

### One-shot ops

Ops jobs (test runs, cache rebuilds, analysis) are still rows — author
a small plan module under `graphids/plan/plans/{smoke,data}/` whose `build()`
emits a single row with the right `action`/`command`/`resources`, then
run the same `graphids run | submit` pair.

### Profiles

Source of truth: `configs/resources/submit_profiles.json`, keyed
`[mode][cluster][length]`. Keys map directly to `SlurmProvider` kwargs
— `submit_row` splats them with `**profile`. `signal_delay_s` is the
one extension: becomes `#SBATCH --signal=USR2@N` so Lightning's
`SLURMEnvironment(auto_requeue=True, requeue_signal=SIGUSR2)` plugin
(wired in `orchestrate._trainer_kwargs`) fires before walltime.

### Preemption auto-resume

Profiles set `signal_delay_s=300` → `--signal=USR2@300`. Five minutes
before walltime, Lightning's `SLURMEnvironment` calls
`scontrol requeue $SLURM_JOB_ID` — same job ID, downstream `afterok`
deps stay valid. (USR2 because NCCL catches USR1.)

## Login node safety

**Safe**: import checks, `graphids run -o plan.json` (pure render), DuckDB, git, ruff.
**SLURM-only**: anything importing torch, instantiating a model, hitting CUDA, pytest, cache rebuilds.

## Data I/O

Jobs read raw CSVs + cache tensors directly from ESS NFS
(`/fs/ess/PAS1266/graphids/{raw,cache}/`). Direct NFS is fast enough for
the working set; if I/O-bound later, add a real staging command —
don't paper over with a silent eval.
