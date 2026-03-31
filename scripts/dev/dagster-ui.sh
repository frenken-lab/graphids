#!/usr/bin/env bash
# Launch dagster-webserver + daemon on login node.
# Access via SSH tunnel: ssh -L 3000:localhost:3000 pitzer.osc.edu
#
# Usage: bash scripts/dev/dagster-ui.sh [port]
#   Ctrl-C stops both processes.
set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
cd "$PROJECT_ROOT"

set -a; source ./.env; set +a
: "${DAGSTER_HOME:?DAGSTER_HOME not set after sourcing .env}"
PORT="${1:-3000}"

module load python/3.12 2>/dev/null || true

cleanup() {
    echo "Stopping dagster processes..."
    kill "$DAEMON_PID" "$WEB_PID" 2>/dev/null
    wait "$DAEMON_PID" "$WEB_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

# Daemon: executes queued runs, handles retries
.venv/bin/dagster-daemon run -m graphids.orchestrate.definitions &
DAEMON_PID=$!
echo "dagster-daemon started (pid=$DAEMON_PID)"

echo "Starting dagster-webserver on port ${PORT}..."
echo "Connect via: ssh -L ${PORT}:localhost:${PORT} pitzer.osc.edu"
echo "Then open: http://localhost:${PORT}"

.venv/bin/dagster-webserver \
    -m graphids.orchestrate.definitions \
    -h 127.0.0.1 \
    -p "$PORT" &
WEB_PID=$!

wait -n
cleanup
