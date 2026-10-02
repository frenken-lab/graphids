#!/usr/bin/env bash
#SBATCH --job-name=cross-vehicle-eval
#SBATCH --partition=gpu
#SBATCH --account=pas1266
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/fs/ess/PAS1266/graphids/slurm_logs/cross-vehicle-eval_%j.out
#SBATCH --error=/fs/ess/PAS1266/graphids/slurm_logs/cross-vehicle-eval_%j.err

set -euo pipefail
source scripts/slurm/_preamble.sh
python -m scripts.cross_vehicle_evaluate "$1" "$2" "$3"
