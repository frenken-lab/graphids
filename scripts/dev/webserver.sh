#!/usr/bin/env bash
# Launch dagster-webserver on login node for asset status UI.
#
# Usage:
#   scripts/dev/webserver.sh [PORT]
#
# Access from local machine:
#   ssh -L PORT:localhost:PORT pitzer.osc.edu
#   open http://localhost:PORT
#
# The webserver reads DAGSTER_HOME (SQLite on scratch) and discovers
# the code location from pyproject.toml [tool.dg]. No torch imports
# at definition time — safe on login nodes.
set -euo pipefail
cd "$(dirname "$0")/../.."

source .venv/bin/activate
source .env

export DAGSTER_HOME="${DAGSTER_HOME:-/fs/scratch/PAS1266/dagster}"
PORT="${1:-3131}"

echo "=== Dagster Webserver ==="
echo "Port:    $PORT"
echo "Home:    $DAGSTER_HOME"
echo "Connect: ssh -L $PORT:localhost:$PORT $(hostname)"
echo "Then:    http://localhost:$PORT"
echo "========================="

exec dg dev -p "$PORT" -h 0.0.0.0
