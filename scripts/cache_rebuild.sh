#!/usr/bin/env bash
# scripts/cache_rebuild.sh — render rebuild_cache plan and submit one CPU
# job per (dataset, vocab_scope) row.
#
# Usage:
#   scripts/cache_rebuild.sh <dataset> [cluster]
#
#   dataset  catalog key (e.g. hcrl_sa, hcrl_ch, set_01)
#   cluster  pitzer | cardinal | ascend  (default: pitzer)
#
# Submits short CPU-mode jobs (vocab scan + windowing — no GPU needed).
# Job ids printed to stdout; plan JSON written to a tempfile.

set -euo pipefail

DATASET="${1:?usage: $0 <dataset> [cluster]}"
CLUSTER="${2:-pitzer}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate
set -a; source ./.env; set +a

PLAN="$(mktemp -t cache_plan_${DATASET}_XXXX.json)"
python -m graphids run data.rebuild_cache --dataset "$DATASET" --seed 42 -o "$PLAN"

echo "submitting cache rebuild for dataset=$DATASET on cluster=$CLUSTER" >&2
jq -c '.rows[]' "$PLAN" | while read -r row; do
    python -m graphids submit --row "$row" --cluster "$CLUSTER" --length short
done
