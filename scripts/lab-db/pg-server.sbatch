#!/usr/bin/env bash
#SBATCH --job-name=kd-pg-server
#SBATCH --partition=cpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --dependency=singleton
#SBATCH --signal=B:SIGINT@60
#SBATCH --output=slurm_logs/%j-pg-server.out
#SBATCH --error=slurm_logs/%j-pg-server.err
# Usage: sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/lab-db/pg-server.sbatch
#
# PGDATA lives on node-local SSD ($TMPDIR) — NFS can't handle postgres mmap.
# State persists via pg_dump to NFS on shutdown + restore on startup.
#
# Key: postgres runs in the FOREGROUND inside `apptainer run ... &`.
# pg_ctl start would daemonize, letting Apptainer exit and unmount the SIF,
# causing SIGBUS when forked children access unmapped library pages.

set -euo pipefail

# NFS-based shared state (survives across jobs)
LAB_DB="/fs/scratch/PAS1266/kd-gat-shared/lab-db"
PG_SECRETS="$LAB_DB/secrets"
PG_LOGS="$LAB_DB/logs"
SIF_PATH="$LAB_DB/postgres-16-bookworm.sif"
ENDPOINT_FILE="$LAB_DB/.pg_endpoint"
DUMP_FILE="$LAB_DB/backup/pg_dumpall.sql.gz"

# Node-local storage (ephemeral, no NFS mmap issues)
PGDATA="${TMPDIR}/kd-pg/pgdata"
PG_RUN="${TMPDIR}/kd-pg/run"

IDLE_TIMEOUT_SECS=7200   # 2 hours
IDLE_CHECK_INTERVAL=900  # 15 minutes

# ── Setup directories ────────────────────────────────────────────────────
mkdir -p "$PGDATA" "$PG_RUN" "$PG_SECRETS" "$PG_LOGS" "$LAB_DB/backup"
chmod 2770 "$LAB_DB"

# ── Build SIF once (cached on NFS) ──────────────────────────────────────
if [[ ! -f "$SIF_PATH" ]]; then
    echo "[pg-server] Building SIF image (one-time)..."
    apptainer build "$SIF_PATH" docker://postgres:16-bookworm
    echo "[pg-server] SIF built: $SIF_PATH"
fi

# ── Password ─────────────────────────────────────────────────────────────
if [[ ! -f "$PG_SECRETS/pgpass" ]]; then
    head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 24 > "$PG_SECRETS/pgpass"
    chmod 640 "$PG_SECRETS/pgpass"
    echo "[pg-server] Generated new password"
fi
PG_PASSWORD=$(cat "$PG_SECRETS/pgpass")

# ── Port selection ───────────────────────────────────────────────────────
find_free_port() {
    local port attempts=0
    while (( attempts < 20 )); do
        port=$(shuf -i 5433-5999 -n1)
        if ! ss -tln | grep -q ":${port} "; then
            echo "$port"
            return 0
        fi
        (( attempts++ ))
    done
    echo "ERROR: Could not find free port after 20 attempts" >&2
    return 1
}

PG_PORT=$(find_free_port)
PG_HOST=$(hostname -f)
echo "[pg-server] PGDATA=$PGDATA  endpoint=$PG_HOST:$PG_PORT"

# Helper for one-shot admin commands (short-lived Apptainer containers — TCP clients only)
pg_exec() {
    apptainer exec \
        --bind "$PGDATA:$PGDATA" \
        --bind "$PG_RUN:$PG_RUN" \
        --env "PGPASSWORD=$PG_PASSWORD" \
        "$SIF_PATH" \
        "$@"
}

# ── Initialize data directory (always — TMPDIR is ephemeral) ─────────────
echo "[pg-server] Running initdb..."
pg_exec initdb -D "$PGDATA" --username=postgres --auth=md5 --pwfile=<(echo "$PG_PASSWORD")

cat > "$PGDATA/pg_hba.conf" <<'EOHBA'
local   all       postgres                  trust
local   all       all                       md5
host    all       all        0.0.0.0/0      md5
host    all       all        ::/0           md5
EOHBA
echo "[pg-server] initdb complete"

# ── Start PostgreSQL in the foreground (keeps Apptainer alive) ───────────
# Use `exec` not `run` to bypass Docker entrypoint (which tries to chmod read-only paths)
apptainer exec \
    --bind "$PGDATA:$PGDATA" \
    --bind "$PG_RUN:$PG_RUN" \
    "$SIF_PATH" \
    postgres -D "$PGDATA" \
        -p "$PG_PORT" \
        -c listen_addresses='*' \
        -c unix_socket_directories="$PG_RUN" \
        -c logging_collector=on \
        -c log_directory="$PG_LOGS" \
        -c log_filename="postgres-${SLURM_JOB_ID}.log" \
    &
PG_PID=$!
echo "[pg-server] postgres PID=$PG_PID"

# ── Wait for readiness ──────────────────────────────────────────────────
echo "[pg-server] Waiting for PostgreSQL..."
for i in $(seq 1 30); do
    if pg_exec pg_isready -h "$PG_HOST" -p "$PG_PORT" -U postgres 2>/dev/null; then
        echo "[pg-server] Ready (took ${i}s)"
        break
    fi
    if ! kill -0 "$PG_PID" 2>/dev/null; then
        echo "[pg-server] ERROR: postgres exited unexpectedly" >&2
        wait "$PG_PID" || true
        exit 1
    fi
    if (( i == 30 )); then
        echo "[pg-server] ERROR: not ready after 30s" >&2
        exit 1
    fi
    sleep 1
done

# ── Create roles and databases (idempotent) ──────────────────────────────
echo "[pg-server] Creating roles and databases..."
pg_exec psql -h "$PG_HOST" -p "$PG_PORT" -U postgres -c \
    "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='kd_writer') THEN CREATE ROLE kd_writer WITH LOGIN PASSWORD '$PG_PASSWORD'; END IF; END \$\$;"

for db in kd_pipeline mlflow; do
    if ! pg_exec psql -h "$PG_HOST" -p "$PG_PORT" -U postgres -tAc \
        "SELECT 1 FROM pg_database WHERE datname='$db'" | grep -q 1; then
        pg_exec psql -h "$PG_HOST" -p "$PG_PORT" -U postgres -c \
            "CREATE DATABASE $db OWNER kd_writer;"
    fi
    pg_exec psql -h "$PG_HOST" -p "$PG_PORT" -U postgres -c \
        "GRANT ALL PRIVILEGES ON DATABASE $db TO kd_writer;"
    pg_exec psql -h "$PG_HOST" -p "$PG_PORT" -U postgres -d "$db" -c \
        "GRANT ALL ON SCHEMA public TO kd_writer;"
done
echo "[pg-server] Roles and databases ready"

# ── Restore from NFS backup if exists ────────────────────────────────────
if [[ -f "$DUMP_FILE" ]]; then
    echo "[pg-server] Restoring from backup..."
    gunzip -c "$DUMP_FILE" | pg_exec psql -h "$PG_HOST" -p "$PG_PORT" -U postgres \
        --set ON_ERROR_STOP=off -q 2>&1 | tail -5
    echo "[pg-server] Restore complete"
fi

# ── Write endpoint file ─────────────────────────────────────────────────
echo "${PG_HOST}:${PG_PORT}" > "$ENDPOINT_FILE"
chmod 640 "$ENDPOINT_FILE"
echo "[pg-server] Endpoint: $(cat "$ENDPOINT_FILE")"

# ── Graceful shutdown: backup then stop ──────────────────────────────────
cleanup() {
    echo "[pg-server] Backing up before shutdown..."
    pg_exec pg_dumpall -h "$PG_HOST" -p "$PG_PORT" -U postgres 2>/dev/null \
        | gzip > "${DUMP_FILE}.tmp" \
        && mv "${DUMP_FILE}.tmp" "$DUMP_FILE" \
        && echo "[pg-server] Backup saved: $(du -h "$DUMP_FILE" | cut -f1)" \
        || echo "[pg-server] WARNING: Backup failed"

    kill -INT "$PG_PID" 2>/dev/null || true
    wait "$PG_PID" 2>/dev/null || true
    rm -f "$ENDPOINT_FILE"
    echo "[pg-server] Shutdown complete"
}
trap cleanup SIGINT SIGTERM EXIT

# ── Idle monitor loop ───────────────────────────────────────────────────
idle_seconds=0
echo "[pg-server] Idle monitor started (timeout=${IDLE_TIMEOUT_SECS}s)"

while kill -0 "$PG_PID" 2>/dev/null; do
    sleep "$IDLE_CHECK_INTERVAL" &
    wait $! 2>/dev/null || break  # interruptible by signals

    active=$(pg_exec psql -h "$PG_HOST" -p "$PG_PORT" -U postgres -tAc \
        "SELECT count(*) FROM pg_stat_activity WHERE datname NOT IN ('postgres','template0','template1') AND pid != pg_backend_pid();" \
        2>/dev/null || echo "0")

    if [[ "$active" == "0" ]]; then
        idle_seconds=$(( idle_seconds + IDLE_CHECK_INTERVAL ))
        echo "[pg-server] Idle: ${idle_seconds}s / ${IDLE_TIMEOUT_SECS}s"
        if (( idle_seconds >= IDLE_TIMEOUT_SECS )); then
            echo "[pg-server] Idle timeout — shutting down"
            break
        fi
    else
        idle_seconds=0
        echo "[pg-server] Active: $active connection(s)"
        # Periodic backup while active
        pg_exec pg_dumpall -h "$PG_HOST" -p "$PG_PORT" -U postgres 2>/dev/null \
            | gzip > "${DUMP_FILE}.tmp" \
            && mv "${DUMP_FILE}.tmp" "$DUMP_FILE" 2>/dev/null || true
    fi
done

# cleanup runs via trap
