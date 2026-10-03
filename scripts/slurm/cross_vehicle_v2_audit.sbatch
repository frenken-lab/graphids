#!/usr/bin/env bash
#SBATCH --job-name=cross-vehicle-v2-audit
#SBATCH --partition=cpu
#SBATCH --account=pas1266
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/fs/ess/PAS1266/graphids/slurm_logs/cross-vehicle-v2-audit_%j.out
#SBATCH --error=/fs/ess/PAS1266/graphids/slurm_logs/cross-vehicle-v2-audit_%j.err

set -euo pipefail
export SKIP_CUDA_CONF=1
source scripts/slurm/_preamble.sh
python -m scripts.cross_vehicle_v2_audit \
  /fs/ess/PAS1266/graphids/dev/rf15/cross_vehicle \
  /fs/ess/PAS1266/graphids/runs/cross_vehicle_v2/comparison.json \
  /fs/ess/PAS1266/graphids/runs/cross_vehicle_v2/audit.json
