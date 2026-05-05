# SLURM

`graphids submit` is the only caller of `parsl.providers.SlurmProvider`.
The sbatch script body is the literal command
`python -m graphids exec --row '<json>' [--ckpt-path X]`, wrapped by
`SrunLauncher`. **No closure pickle** — code fixes committed after
submission DO reach a pending job, since the job re-imports current
source at exec time.

Profiles in `configs/resources/submit_profiles.json`
(`[mode][cluster][length]`) hold Parsl `SlurmProvider` kwargs
verbatim — `submit_row` splats them with `**profile`. Preempt-resume
runs via SIGUSR2 (USR2 because NCCL catches USR1) — Lightning's
`SLURMEnvironment(auto_requeue=True, requeue_signal=SIGUSR2)` plugin
(wired in `graphids.orchestrate._trainer_kwargs`) calls
`scontrol requeue $SLURM_JOB_ID`, same job ID, downstream `afterok`
chains stay valid.

`graphids.slurm.sizing` (MLflow-history walltime estimation) was
removed in the 2026-05-01 chassis rebuild — use the static
per-length defaults in `submit_profiles.json`.

## `graphids.slurm`

::: graphids.slurm
    options:
      show_submodules: true
