# KD-GAT SLURM / HPC Conventions

## Environment

- **Cluster**: OSC Pitzer (Ohio Supercomputer Center), RHEL 9, SLURM
- **GPU**: 2x V100 per node, ~362 GB RAM (account from `$KD_GAT_SLURM_ACCOUNT` in `.env`, gpu partition)
- **Python**: 3.12 via `module load python/3.12`, uv venv `.venv/`
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)

## Rules

- Spawn/fork CUDA rule: See critical-constraints.md.
- Test on small datasets (`hcrl_ch`) before large ones (`set_02`+).
- SLURM logs go to `slurm_logs/`, experiment outputs to `experimentruns/`.
- Heavy tests use `@pytest.mark.slurm` — auto-skipped on login nodes.
- **Always run tests via SLURM** (`cpu` partition, 8 CPUs, 16GB). Submit with `bash scripts/slurm/run_tests_slurm.sh`.

## Login Node Safety

**Safe on login node:**
- Import checks: `python -c "from graphids.config import resolve; print('OK')"`
- Exports: `python -m graphids.pipeline.export`
- DuckDB rebuild: `python -m graphids.pipeline.build_analytics`
- Quarto: `quarto render`, `quarto preview`
- Git, DVC, W&B sync, ruff

**Must go through SLURM:**
- `python -m graphids.pipeline.cli <any stage>` — all training/evaluation
- `python -m pytest` — test suite
- Any script that imports and runs models
