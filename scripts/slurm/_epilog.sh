#!/usr/bin/env bash
# scripts/slurm/_epilog.sh — sourced at end of SLURM job scripts.
# GPU utilization is logged by Lightning's DeviceStatsMonitor callback.
# This script handles SLURM accounting and log hygiene only.

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo ""
    echo "=== SLURM Accounting ==="
    sacct -j "$SLURM_JOB_ID" --format=JobID%15,Elapsed,MaxRSS%12,MaxVMSize%12 \
        --noheader 2>/dev/null || echo "  (sacct not available)"
fi

# Rotate old logs (30-day retention)
find "${PROJECT_ROOT:-/users/PAS2022/rf15/KD-GAT}/slurm_logs/" \
    \( -name "*.out" -o -name "*.err" \) -mtime +30 -delete 2>/dev/null || true
