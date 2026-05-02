# SLURM

`graphids submit` is the only caller of `parsl.providers.SlurmProvider`.
The sbatch script body is the literal command
`python -m graphids exec --row '<json>' [--ckpt-path X]`, wrapped by
`SrunLauncher`. **No closure pickle** — code fixes committed after
submission DO reach a pending job, since the job re-imports current
source at exec time.

Profiles in `configs/resources/submit_profiles.json`
(`[mode][cluster][length]`) translate to Parsl `SlurmProvider`
kwargs. Preempt-resume runs via SIGUSR2 (USR2 because NCCL catches
USR1) — `graphids/runtime.py` traps the signal and re-submits the
row with `--ckpt-path={run_dir}/checkpoints/last.ckpt` and
`--depends-on-afterany=$SLURM_JOB_ID`.

`graphids.slurm.sizing` (MLflow-history walltime estimation) was
removed in the 2026-05-01 chassis rebuild — use the static
per-length defaults in `submit_profiles.json`.

## `graphids.slurm`

::: graphids.slurm
    options:
      show_submodules: true
