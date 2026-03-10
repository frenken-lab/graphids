#!/usr/bin/env bash
# reproduce.sh — Run the full KD-GAT pipeline for paper results.
#
# Submits all dataset × seed combinations via SLURM. Each job runs the
# complete 3-stage pipeline (VGAE → GAT → DQN) with Ray orchestration.
#
# Usage:
#   bash scripts/reproduce.sh              # Submit all jobs
#   bash scripts/reproduce.sh --dry-run    # Print sbatch commands without submitting
set -euo pipefail

DATASETS=(hcrl_ch hcrl_sa set_01 set_02 set_03 set_04)
SEEDS=(42 123 456 789 1024)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm/ray_slurm.sbatch"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "ERROR: SLURM script not found: $SLURM_SCRIPT" >&2
    exit 1
fi

total=0
for seed in "${SEEDS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        cmd="sbatch ${SLURM_SCRIPT} flow --dataset ${ds} -O training.seed ${seed}"
        if $DRY_RUN; then
            echo "[dry-run] $cmd"
        else
            echo "Submitting: dataset=${ds} seed=${seed}"
            $cmd
        fi
        ((total++))
    done
done

echo ""
echo "Total jobs: ${total} (${#DATASETS[@]} datasets × ${#SEEDS[@]} seeds)"
if $DRY_RUN; then
    echo "(dry-run mode — no jobs were submitted)"
fi
