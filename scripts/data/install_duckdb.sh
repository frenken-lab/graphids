#!/usr/bin/env bash
# Install DuckDB CLI to the shared tools directory.
# DuckDB is not available as an OSC module, so we bundle the binary.
#
# Usage: bash scripts/data/install_duckdb.sh

set -euo pipefail

SHARED="${KD_GAT_SHARED_ROOT:?Set KD_GAT_SHARED_ROOT in .env}"
TOOLS_DIR="$SHARED/tools"
mkdir -p "$TOOLS_DIR"

echo "Downloading DuckDB CLI..."
curl -fSL "https://github.com/duckdb/duckdb/releases/latest/download/duckdb_cli-linux-amd64.zip" \
    -o /tmp/duckdb_cli.zip

echo "Extracting to $TOOLS_DIR..."
unzip -o /tmp/duckdb_cli.zip -d "$TOOLS_DIR/"
chmod +x "$TOOLS_DIR/duckdb"
rm /tmp/duckdb_cli.zip

echo "DuckDB installed:"
"$TOOLS_DIR/duckdb" --version

echo ""
echo "Usage: $TOOLS_DIR/duckdb $SHARED/data/datalake/analytics.duckdb"
