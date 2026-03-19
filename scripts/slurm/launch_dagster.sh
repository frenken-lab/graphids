#!/usr/bin/env bash
# scripts/slurm/launch_dagster.sh — Submit dagster daemon and print SSH tunnel command.
#
# Usage:
#   bash scripts/slurm/launch_dagster.sh               # Submit daemon
#   bash scripts/slurm/launch_dagster.sh --resubmit     # Enable auto-resubmit before timeout
#   bash scripts/slurm/launch_dagster.sh --time 48:00:00 # Override walltime
#   bash scripts/slurm/launch_dagster.sh --stop          # Cancel running daemon

set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
SBATCH_SCRIPT="$PROJECT_ROOT/scripts/slurm/dagster_daemon.sbatch"
CONNECTION_FILE="$PROJECT_ROOT/.dagster/connection_info.txt"

# --- Parse arguments ---
RESUBMIT=0
STOP=0
TIME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resubmit) RESUBMIT=1; shift ;;
        --stop) STOP=1; shift ;;
        --time) TIME_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Check for existing daemon ---
EXISTING_JOB=$(squeue -u "$USER" -n dagster-daemon -h -o "%i" 2>/dev/null | head -1)

if [[ "$STOP" == "1" ]]; then
    if [[ -n "$EXISTING_JOB" ]]; then
        echo "Cancelling dagster daemon job $EXISTING_JOB..."
        scancel "$EXISTING_JOB"
        rm -f "$CONNECTION_FILE"
        echo "Done."
    else
        echo "No dagster daemon job found."
    fi
    exit 0
fi

if [[ -n "$EXISTING_JOB" ]]; then
    echo "Dagster daemon already running: job $EXISTING_JOB"
    if [[ -f "$CONNECTION_FILE" ]]; then
        echo ""
        cat "$CONNECTION_FILE"
        echo ""
        TUNNEL_CMD=$(grep "^tunnel_cmd=" "$CONNECTION_FILE" | cut -d= -f2-)
        echo "SSH tunnel: $TUNNEL_CMD"
    fi
    exit 0
fi

# --- Build sbatch command ---
SBATCH_ARGS=()
if [[ -n "$TIME_OVERRIDE" ]]; then
    SBATCH_ARGS+=(--time "$TIME_OVERRIDE")
fi

EXPORT_VARS=""
if [[ "$RESUBMIT" == "1" ]]; then
    EXPORT_VARS="KD_GAT_DAGSTER_RESUBMIT=1"
fi

if [[ -n "$EXPORT_VARS" ]]; then
    SBATCH_ARGS+=(--export "ALL,$EXPORT_VARS")
fi

# --- Submit ---
echo "Submitting dagster daemon..."
JOB_ID=$(sbatch --parsable "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT")
echo "Submitted job $JOB_ID"

# --- Wait for RUNNING state (max 5 min) ---
echo "Waiting for job to start..."
MAX_WAIT=300
ELAPSED=0
INTERVAL=5

while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "UNKNOWN")
    if [[ "$STATE" == "RUNNING" ]]; then
        break
    elif [[ "$STATE" == "UNKNOWN" || "$STATE" == "FAILED" || "$STATE" == "CANCELLED" ]]; then
        echo "Job $JOB_ID is in state $STATE — aborting."
        exit 1
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    echo "  State: $STATE ($ELAPSED/${MAX_WAIT}s)"
done

if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    echo "Job did not start within ${MAX_WAIT}s. Check: squeue -j $JOB_ID"
    exit 1
fi

# --- Wait briefly for connection info file ---
sleep 5

if [[ -f "$CONNECTION_FILE" ]]; then
    echo ""
    echo "=== Dagster Daemon Running ==="
    cat "$CONNECTION_FILE"
    echo ""
    TUNNEL_CMD=$(grep "^tunnel_cmd=" "$CONNECTION_FILE" | cut -d= -f2-)
    echo "On your local machine, run:"
    echo "  $TUNNEL_CMD"
    echo ""
    PORT=$(grep "^port=" "$CONNECTION_FILE" | cut -d= -f2-)
    echo "Then open: http://localhost:$PORT"
else
    echo ""
    echo "Job is running but connection info not yet written."
    echo "Check: cat $CONNECTION_FILE"
fi
