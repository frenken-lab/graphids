#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=PAS1266
#SBATCH --output=/fs/ess/PAS1266/kd-gat/slurm_logs/fusion_cpu_%x_%j.out
#SBATCH --error=/fs/ess/PAS1266/kd-gat/slurm_logs/fusion_cpu_%x_%j.err

# Usage: sbatch --job-name=<name> run_fusion_cpu.sh <method> <scale> <run_dir>
# Example:
#   sbatch --job-name=fusion_bandit_small run_fusion_cpu.sh bandit small \
#     /fs/ess/PAS1266/kd-gat/dev/rf15/set_01/fusion_small_fusion_82437173/seed_42

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 source "$SCRIPT_DIR/_preamble.sh"

METHOD="${1:?Usage: $0 <method> <scale> <run_dir>}"
SCALE="${2:?Usage: $0 <method> <scale> <run_dir>}"
RUN_DIR="${3:?Usage: $0 <method> <scale> <run_dir>}"

# CPU parallelism — use all allocated cores for BLAS/MKL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

LAKE=/fs/ess/PAS1266/kd-gat/dev/rf15/set_01

# Map scale to cached states dir
case "$SCALE" in
    small) STATES_DIR="$LAKE/fusion_states/small_ff9f9014_bf2a5575" ;;
    large) STATES_DIR="$LAKE/fusion_states/large_9ffb88b1_e9354ccd" ;;
    *) echo "Unknown scale: $SCALE"; exit 1 ;;
esac

echo "=== Fusion CPU training: method=$METHOD scale=$SCALE cpus=$SLURM_CPUS_PER_TASK ==="
echo "Cached states: $STATES_DIR"
echo "Run dir: $RUN_DIR"

python -m graphids fit \
    --config graphids/config/stages/fusion.yaml \
    --config "graphids/config/fusion/base.yaml" \
    --config "graphids/config/fusion/methods/${METHOD}.yaml" \
    --config "graphids/config/fusion/scales/${SCALE}.yaml" \
    --data.init_args.cached_states_dir="$STATES_DIR" \
    --data.init_args.dataset=set_01 \
    --seed_everything=42 \
    --trainer.default_root_dir="$RUN_DIR"

source "$SCRIPT_DIR/_epilog.sh"
