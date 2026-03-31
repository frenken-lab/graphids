#!/bin/bash
# One-shot: wait for in-flight GPU jobs from cancelled dagster run,
# touch .complete markers, then resubmit dagster orchestrator.
# Run in background: nohup bash scripts/slurm/_wait_and_resubmit.sh &
set -euo pipefail

LAKE=/fs/ess/PAS1266/kd-gat/dev/rf15/set_01

JOBS=(46152810 46154697)
DIRS=(
    "${LAKE}/vgae_large_autoencoder_bf355e79/seed_42"
    "${LAKE}/gat_small_curriculum_b2d0042f/seed_42"
)

echo "[$(date)] Waiting for ${#JOBS[@]} jobs: ${JOBS[*]}"

for i in "${!JOBS[@]}"; do
    jid="${JOBS[$i]}"
    rd="${DIRS[$i]}"

    while true; do
        state=$(sacct -j "$jid" --format=State --noheader -P | head -1)
        case "$state" in
            RUNNING|PENDING|COMPLETING)
                sleep 120
                ;;
            COMPLETED)
                echo "[$(date)] Job $jid COMPLETED — touching ${rd}/.complete"
                touch "${rd}/.complete"
                break
                ;;
            *)
                echo "[$(date)] Job $jid ended with state=$state — NOT marking complete"
                break
                ;;
        esac
    done
done

echo "[$(date)] Submitting new dagster ablation run"
cd /users/PAS2022/rf15/KD-GAT
sbatch scripts/slurm/run_ablation.sh
echo "[$(date)] Done"
