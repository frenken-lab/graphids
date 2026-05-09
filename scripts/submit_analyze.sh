#!/usr/bin/env bash
set -euo pipefail
source .env
source .venv/bin/activate

for ds in hcrl_sa set_01 set_02 set_03 set_04; do
    plan="rendered/${ds}/ops/analyze/seed_42.json"
    mkdir -p "$(dirname "$plan")"
    gx run ops.analyze -d "$ds" -s 42 -o "$plan"
    gx plans submit --plan "$plan" -C pitzer
done
