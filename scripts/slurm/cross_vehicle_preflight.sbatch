#!/usr/bin/env bash
#SBATCH --job-name=cross-vehicle-preflight-v2
#SBATCH --partition=gpu
#SBATCH --account=pas1266
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/fs/ess/PAS1266/graphids/slurm_logs/cross-vehicle-preflight-v2_%j.out
#SBATCH --error=/fs/ess/PAS1266/graphids/slurm_logs/cross-vehicle-preflight-v2_%j.err

set -euo pipefail
source scripts/slurm/_preamble.sh
python -m scripts.cross_vehicle_preflight \
  /fs/ess/PAS1266/graphids/runs/cross_vehicle_v2/preflight_nograph.json \
  configs/experiments/cross_vehicle/silverado_raw_fit.yml \
  configs/experiments/cross_vehicle/silverado_behavioral_fit.yml
