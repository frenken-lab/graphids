#!/usr/bin/env bash
# scripts/slurm/_epilog.sh — sourced at end of GPU job scripts.
# Prints GPU utilization report for resource right-sizing.
#
# Expects:
#   SLURM_JOB_ID — set by SLURM

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
