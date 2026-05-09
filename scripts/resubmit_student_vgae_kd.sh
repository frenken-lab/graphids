#!/usr/bin/env bash
set -euo pipefail
source .env
source .venv/bin/activate

for ds in hcrl_sa set_01 set_02 set_03 set_04; do
    PLAN="rendered/${ds}/training/main/seed_42.json"
    jid=$(gx submit --plan "$PLAN" --row-name student_vgae_kd --cluster pitzer)
    echo "${ds}: ${jid}"
done
