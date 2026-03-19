#!/bin/bash
#SBATCH --job-name=dagster-daemon
#SBATCH --partition=cpu
#SBATCH --account=PAS1266
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=slurm_logs/dagster-daemon-%j.out
#SBATCH --error=slurm_logs/dagster-daemon-%j.out
#
# Dagster daemon + webserver as a CPU-only SLURM job.
# Child GPU jobs are submitted via sbatch from within the daemon.
# Access UI via SSH port forwarding.

SKIP_STAGE_DATA=1 SKIP_CUDA_CONF=1 source scripts/slurm/_preamble.sh

log_job_header "Dagster Daemon"

# --- Port assignment ---
# Deterministic per-user port to avoid collisions on shared nodes
DAGSTER_PORT=$((3000 + ($(id -u) % 1000)))

# --- Dagster home ---
export DAGSTER_HOME="$PROJECT_ROOT/.dagster"

# --- Connection info ---
NODE=$(hostname -s)
CONNECTION_FILE="$DAGSTER_HOME/connection_info.txt"
cat > "$CONNECTION_FILE" <<EOF
job_id=$SLURM_JOB_ID
node=$NODE
port=$DAGSTER_PORT
url=http://$NODE:$DAGSTER_PORT
tunnel_cmd=ssh -L $DAGSTER_PORT:$NODE:$DAGSTER_PORT $(whoami)@pitzer.osc.edu
started=$(date -Iseconds)
EOF

echo "Connection info written to $CONNECTION_FILE"
echo ""
echo "=== Access Instructions ==="
echo "From your local machine, run:"
echo "  ssh -L $DAGSTER_PORT:$NODE:$DAGSTER_PORT $(whoami)@pitzer.osc.edu"
echo "Then open: http://localhost:$DAGSTER_PORT"
echo ""

# --- Graceful shutdown on SIGUSR1 (SLURM pre-timeout signal) ---
handle_dagster_shutdown() {
    echo "[$(date)] SIGUSR1 received — shutting down Dagster..."
    if [[ -n "${DAGSTER_PID:-}" ]]; then
        kill -TERM "$DAGSTER_PID" 2>/dev/null || true
        wait "$DAGSTER_PID" 2>/dev/null
    fi

    # Optional self-resubmit for long-running sessions
    if [[ "${KD_GAT_DAGSTER_RESUBMIT:-0}" == "1" ]]; then
        echo "[$(date)] Resubmitting dagster daemon job..."
        NEW_JOB_ID=$(sbatch --parsable "$PROJECT_ROOT/scripts/slurm/dagster_daemon.sbatch")
        echo "[$(date)] Resubmitted as job $NEW_JOB_ID"
    fi

    # Clean up connection info
    rm -f "$CONNECTION_FILE"
    log_job_footer 0
    exit 0
}
trap handle_dagster_shutdown USR1

# --- Launch Dagster ---
echo "[$(date)] Starting dagster dev on port $DAGSTER_PORT..."
dagster dev -w "$PROJECT_ROOT/workspace.yaml" -p "$DAGSTER_PORT" &
DAGSTER_PID=$!

# Wait for the process (allows trap to fire)
wait "$DAGSTER_PID"
EXIT_CODE=$?

rm -f "$CONNECTION_FILE"
log_job_footer $EXIT_CODE
exit $EXIT_CODE
