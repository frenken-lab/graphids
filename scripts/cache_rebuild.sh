#!/usr/bin/env bash
# scripts/cache_rebuild.sh — render rebuild_cache plan and submit each row.
#
# Thin wrapper around `gx run | gx plans submit`. Both rows (voc=train,
# voc=all) are short CPU jobs (vocab scan + windowing — no GPU needed).
# Job ids printed by `gx plans submit` per row.
#
# Usage:
#   scripts/cache_rebuild.sh <dataset> [cluster]
#
#   dataset  catalog key (e.g. hcrl_sa, hcrl_ch, set_01)
#   cluster  pitzer | cardinal | ascend  (default: pitzer)

set -euo pipefail

DATASET="${1:?usage: $0 <dataset> [cluster]}"
CLUSTER="${2:-pitzer}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate
set -a; source ./.env; set +a

PLAN="$(mktemp -t cache_plan_${DATASET}_XXXX.json)"
gx run data.rebuild_cache --dataset "$DATASET" --seed 42 -o "$PLAN"
gx plans submit --plan "$PLAN" --cluster "$CLUSTER" --length short
