# GraphIDS SLURM / HPC Conventions

## Environment

- **Cluster**: OSC Pitzer (Ohio Supercomputer Center), RHEL 9, SLURM
- **Account**: PAS1266 (`$GRAPHIDS_SLURM_ACCOUNT` in `.env`). Must `source .env` before submitting on a login node.
- **GPU**: Pitzer 2x V100 / Cardinal H100 / Ascend A100. See `~/.claude/projects/-users-PAS2022-rf15-graphids/memory/reference_osc_gpu_clusters.md`.
- **Python**: 3.12 via `module load python/3.12`, uv venv `.venv/`
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)

## Rules

- Spawn/fork CUDA rule: see `critical-constraints.md`.
- Test on small datasets (`hcrl_sa`) before large ones (`set_02`+).
- SLURM logs go to `slurm_logs/`; experiment outputs to `{RUN_ROOT}/...`.
- Heavy tests use `@pytest.mark.slurm` — auto-skipped on login nodes.
- **Never run `pytest` on login nodes.** Submit a one-row ops job (see below).

## Job Submission — four-step chassis

The CLI surface is `graphids run | exec | submit`. There is no
`python -m graphids fit/test`, no submitit, no `--mode` / `--command`
ops shortcut — every job is a row. See
`.claude/rules/single-submission-primitive.md`.

```bash
# Render + validate one plan into a JSON array.
python -m graphids run configs/plans/ofat.jsonnet --dataset hcrl_sa --seed 42 -o plan.json

# Submit each row to SLURM. graphids submit is the ONLY caller of
# parsl.providers.SlurmProvider.submit. Prints jid on stdout.
jq -c '.[]' plan.json | while read row; do
    python -m graphids submit --row "$row" --cluster pitzer --length long
done

# Same-batch deps (afterok = data dep, afterany = preempt-resume chain).
python -m graphids submit --row "$row" --cluster cardinal \
    --depends-on-afterok 12345 --ckpt-path /path/to/upstream/best.ckpt
```

The sbatch script body is a literal bash string:
`python -m graphids exec --row '<json>' [--ckpt-path X]`. Wrapped by
`SrunLauncher`. **No pickle** of Python closures — code fixes committed
after submission DO reach a pending job, because the job re-imports the
current source at exec time. (Contrast: the old submitit path pickled
the entire closure and was vulnerable to source drift.)

### One-shot ops

Ops jobs (test runs, cache rebuilds, analysis) are still rows — author
a small plan jsonnet under `configs/plans/ops/` that emits a single row
with the right `action` / `command` / `resources`, then run the same
`graphids run | submit` pair. Resist adding a `--mode/--command` ops
shortcut to `submit`; that's exactly the multi-shape entry point this
chassis avoids.

| Job             | Where it lives                                                |
|-----------------|---------------------------------------------------------------|
| Tests           | one-row plan invoking `python -m pytest [-k pattern]`         |
| Cache rebuild   | one-row plan invoking `python -m graphids data rebuild ...`   |
| Analyze ckpt    | one-row plan invoking `python -m graphids analyze ...`        |

### Profiles

Source of truth for `partition` / `cores_per_node` / `mem_per_node` /
`walltime` / `gpus_per_node` / `signal_delay_s`:
`configs/resources/submit_profiles.json`, keyed `[mode][cluster][length]`.
Keys map directly to `parsl.providers.SlurmProvider` kwargs — `submit_row`
splats them with `**profile`, no translation layer. `signal_delay_s` is
the one extension: it becomes `#SBATCH --signal=USR2@N` in
`scheduler_options` so the SIGUSR2 trap in `runtime.py` fires before
walltime.

### Preemption auto-resume

Profiles set `signal_delay_s=300`, so Parsl's sbatch script emits
`--signal=USR2@300`. The SIGUSR2 trap lives in `graphids/runtime.py` —
five minutes before walltime, the process re-submits the row with
`--ckpt-path={run_dir}/checkpoints/last.ckpt` and
`--depends-on-afterany=$SLURM_JOB_ID`. No manual resubmit loop. (USR2
because NCCL catches USR1.)

## Data I/O

Jobs read raw CSVs and cache tensors directly from ESS NFS
(`/fs/ess/PAS1266/graphids/{raw,cache}/`). The old `stage-data` command
(NFS → scratch → TMPDIR) was removed 2026-04-14 after rebuild confirmed
direct NFS reads are fast enough for our working set. If training ever
becomes I/O-bound, reintroduce a real staging command — don't paper over
with a silent eval.

## Login Node Safety

**Safe on login node:**
- Import checks: `python -c "from graphids.blueprint import TrainRow; print('OK')"`
- `graphids run <plan> -o plan.json` (pure render — no torch import)
- `graphids exec --row <tiny-cpu-row>` for quick CPU smoke (rare; prefer SLURM)
- DuckDB queries; git; ruff

**Must go through SLURM (`graphids submit`):**
- Anything that imports torch / instantiates a model / hits CUDA
- The pytest suite
- Cache rebuilds and dataset preprocessing
