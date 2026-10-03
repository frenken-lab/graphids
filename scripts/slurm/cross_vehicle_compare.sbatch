#!/usr/bin/env bash
#SBATCH --job-name=cross-vehicle-compare
#SBATCH --partition=cpu
#SBATCH --account=pas1266
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/fs/ess/PAS1266/graphids/slurm_logs/cross-vehicle-compare_%j.out
#SBATCH --error=/fs/ess/PAS1266/graphids/slurm_logs/cross-vehicle-compare_%j.err

set -euo pipefail
export SKIP_CUDA_CONF=1
source scripts/slurm/_preamble.sh
python -m scripts.cross_vehicle_compare "$@"
