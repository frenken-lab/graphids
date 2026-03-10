#!/usr/bin/env bash
# scripts/lab-db/ensure_pg.sh — Sourceable launcher for shared PostgreSQL.
#
# Ensures a PostgreSQL SLURM job is running, waits for readiness, then
# exports KD_GAT_DB_URI and MLFLOW_TRACKING_URI.
#
# Usage:
#   source scripts/lab-db/ensure_pg.sh
#   echo "$KD_GAT_DB_URI"           # postgresql://kd_writer:***@host:port/kd_pipeline
#   echo "$MLFLOW_TRACKING_URI"     # postgresql://kd_writer:***@host:port/mlflow

# Don't set -e since this is sourced (would kill the caller's shell)

_ENSURE_PG_RC=0

_ensure_pg() {
    local PROJECT_ROOT="${PROJECT_ROOT:-/users/PAS2022/rf15/KD-GAT}"
    local LAB_DB="/fs/scratch/PAS1266/kd-gat-shared/lab-db"
    local ENDPOINT_FILE="$LAB_DB/.pg_endpoint"
    local SECRETS_FILE="$LAB_DB/secrets/pgpass"
    local TIMEOUT=90  # seconds to wait for readiness
    local JOB_NAME="kd-pg-server"

    # Check if already running (any user in the group)
    local running
    running=$(squeue --name="$JOB_NAME" -h -o "%T" 2>/dev/null | head -1)

    if [[ -z "$running" ]]; then
        echo "[ensure_pg] No running $JOB_NAME job found — submitting..."
        local account="${KD_GAT_SLURM_ACCOUNT:-PAS1266}"
        sbatch --account="$account" "$PROJECT_ROOT/scripts/lab-db/pg-server.sbatch" >/dev/null 2>&1
        if [[ $? -ne 0 ]]; then
            echo "[ensure_pg] ERROR: sbatch submission failed" >&2
            return 1
        fi
        echo "[ensure_pg] Submitted — waiting for startup..."
    else
        echo "[ensure_pg] Found running $JOB_NAME job (state=$running)"
    fi

    # Wait for endpoint file to appear and be non-empty
    local elapsed=0
    while (( elapsed < TIMEOUT )); do
        if [[ -s "$ENDPOINT_FILE" ]]; then
            break
        fi
        sleep 2
        (( elapsed += 2 ))
    done

    if [[ ! -s "$ENDPOINT_FILE" ]]; then
        echo "[ensure_pg] ERROR: Endpoint file not ready after ${TIMEOUT}s" >&2
        return 1
    fi

    local endpoint
    endpoint=$(cat "$ENDPOINT_FILE")
    local pg_host="${endpoint%%:*}"
    local pg_port="${endpoint##*:}"

    # Verify TCP connectivity (pg_isready on login node has broken libldap;
    # the endpoint file is only written after in-container pg_isready succeeds,
    # so a simple TCP check is sufficient here)
    if ! bash -c "echo >/dev/tcp/$pg_host/$pg_port" 2>/dev/null; then
        echo "[ensure_pg] WARNING: TCP check failed for $pg_host:$pg_port (may still be starting)" >&2
    fi

    # Read password
    if [[ ! -f "$SECRETS_FILE" ]]; then
        echo "[ensure_pg] ERROR: Password file not found: $SECRETS_FILE" >&2
        return 1
    fi
    local pg_pass
    pg_pass=$(cat "$SECRETS_FILE")

    # Export connection URIs
    export KD_GAT_DB_URI="postgresql://kd_writer:${pg_pass}@${pg_host}:${pg_port}/kd_pipeline"
    export MLFLOW_TRACKING_URI="postgresql://kd_writer:${pg_pass}@${pg_host}:${pg_port}/mlflow"

    echo "[ensure_pg] Connected: $pg_host:$pg_port"
    return 0
}

_ensure_pg
_ENSURE_PG_RC=$?
unset -f _ensure_pg
return $_ENSURE_PG_RC 2>/dev/null || exit $_ENSURE_PG_RC
