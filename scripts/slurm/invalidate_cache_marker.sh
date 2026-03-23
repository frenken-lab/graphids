#!/usr/bin/env bash
# Invalidate scratch staging marker after cache rebuild completes.
# Submit with --dependency=afterany:<array_jobid> so it runs after all tasks finish.
#
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --job-name=kd-gat-cache-cleanup
#SBATCH --output=slurm_logs/cache_cleanup_%j.out
#SBATCH --error=slurm_logs/cache_cleanup_%j.err

MARKER="/fs/scratch/PAS1266/kd-gat-data/cache/.staged_marker"

if [ -f "$MARKER" ]; then
    rm "$MARKER"
    echo "Invalidated scratch staging marker (will re-stage on next GPU job)"
else
    echo "No staging marker found — nothing to invalidate"
fi
