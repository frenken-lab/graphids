#!/usr/bin/env bash
#SBATCH --job-name=can-label
#SBATCH --partition=cpu
#SBATCH --account=pas1266
#SBATCH --array=0-19%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/fs/ess/PAS1266/graphids/slurm_logs/can-label_%A_%a.out
#SBATCH --error=/fs/ess/PAS1266/graphids/slurm_logs/can-label_%A_%a.err

set -euo pipefail
cd /users/PAS2022/rf15/projects/lab/graphids
module load python/3.12
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
: "${PHASE:?submit with --export=ALL,PHASE=prepare or PHASE=rank}"
case "${PHASE}" in
  prepare|rank) ;;
  *) echo "PHASE must be prepare or rank" >&2; exit 2 ;;
esac
python -m scripts.run_can_label_completion "${PHASE}" \
  docs/empirical-notes/2026-10-06-signal-catalog/label-completion-manifest-v5-stage.json \
  docs/empirical-notes/2026-10-06-signal-catalog/label-completion-manifest-v2.json \
  "${CAN_LABEL_OUTPUT_DIR:?set CAN_LABEL_OUTPUT_DIR outside the repository}" \
  --case-index "${SLURM_ARRAY_TASK_ID}"
