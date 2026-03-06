#!/usr/bin/env bash
# scripts/slurm/_epilog.sh — sourced at end of GPU job scripts.
# Handles S3 sync, W&B sync, and GPU utilization report.
#
# Expects:
#   SLURM_JOB_ID, SLURM_JOB_NAME — set by SLURM
#   JOB_LOG_PREFIX — set by caller (e.g. "ray", "tune", "sweep_pipeline")

JOB_LOG_PREFIX="${JOB_LOG_PREFIX:-${SLURM_JOB_NAME:-unknown}}"

# --- S3 sync (datalake + sweep results/state) ---
if command -v aws &>/dev/null; then
    for subdir in datalake sweep_results sweep_state; do
        if [[ -d "data/$subdir" ]]; then
            echo "Syncing data/$subdir to S3..."
            aws s3 sync "data/$subdir/" \
                "s3://${KD_GAT_S3_BUCKET:-kd-gat}/$subdir/" \
                2>/dev/null || true
        fi
    done
fi

# --- W&B sync (only runs from this job) ---
if command -v wandb &>/dev/null && [[ -d wandb/ ]]; then
    echo "Syncing offline W&B runs..."
    find wandb/ -maxdepth 1 -name "run-*" \
        -newer "slurm_logs/${JOB_LOG_PREFIX}_${SLURM_JOB_ID}.out" \
        -exec wandb sync {} \; 2>/dev/null || true
fi

# --- GPU utilization report ---
echo ""
echo "=== GPU Utilization Report ==="
if command -v nvidia-smi &>/dev/null; then
    echo "GPU(s):"
    nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi query failed)"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo ""
    echo "SLURM accounting:"
    sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,TRESUsageInTot%80 \
        --noheader 2>/dev/null || echo "  (sacct not available)"
fi
echo "=== End Report ==="
