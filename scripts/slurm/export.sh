#!/usr/bin/env bash
# Export experiment data to reports/data/ for Quarto site.
# CPU-only job — scans MLflow + experimentruns/ filesystem.
#
# Usage:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/export.sh
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/export.sh --reports
#
# --reports: ensure reports/data/ is ready for Quarto preview

#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=kd-gat-export
#SBATCH --output=slurm_logs/export_%j.out
#SBATCH --error=slurm_logs/export_%j.err

SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh"

echo "=== Export Pipeline ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Args:   ${*:-<none>}"
echo ""

python -m graphids.pipeline.export "$@"

EXIT_CODE=$?

echo ""
echo "=== Export $([ $EXIT_CODE -eq 0 ] && echo 'COMPLETE' || echo "FAILED (exit $EXIT_CODE)") ==="

exit $EXIT_CODE
