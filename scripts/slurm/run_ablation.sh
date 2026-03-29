#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --account=PAS1266
#SBATCH --job-name=dagster-ablation
#SBATCH --output=slurm_logs/dagster_%j.out
#SBATCH --signal=B:USR1@300
set -euo pipefail

# --- CPU orchestrator job for Ablation Run 004 ---
# Dagster materializes all 32 assets per partition (dataset|seed).
# Each asset submits its own GPU sbatch job, polls sacct, handles retry.
# Restart-safe: re-running skips completed stages (best_model.ckpt check).
#
# Usage:
#   sbatch scripts/slurm/run_ablation.sh
#   sbatch scripts/slurm/run_ablation.sh --dataset set_01   # single dataset
#   KD_GAT_DRY_RUN=1 sbatch scripts/slurm/run_ablation.sh  # dry-run

SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 \
    source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh"

DATASETS="${KD_GAT_DATASETS:-set_01 set_02}"
SEEDS="${KD_GAT_SEEDS:-42}"

# Parse --dataset flag for single-dataset runs
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASETS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=== Ablation Run 004 ==="
echo "Datasets: ${DATASETS}"
echo "Seeds: ${SEEDS}"
echo "Dry run: ${KD_GAT_DRY_RUN:-false}"
echo "========================"

for dataset in ${DATASETS}; do
    for seed in ${SEEDS}; do
        echo ""
        echo ">>> Materializing: ${dataset}|${seed}"
        python -m graphids.orchestrate --partition "${dataset}|${seed}"
        echo "<<< Done: ${dataset}|${seed}"
    done
done

echo ""
echo "=== All partitions complete ==="
