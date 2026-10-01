#!/usr/bin/env bash
#SBATCH --clusters=pitzer
#SBATCH --account=pas1266
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=temporal-coverage-epoch

set -euo pipefail
cd "${COMPARISON_CODE_ROOT:-/users/PAS2022/rf15/projects/lab/graphids}"
source scripts/slurm/_preamble.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${COMPARISON_ROOT:?set COMPARISON_ROOT}"
: "${COMPARISON_MODE:?set continuous or segmented}"
: "${COMPARISON_LANES:?set lane count}"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA correctness must run on a GPU"'
mkdir -p "$COMPARISON_ROOT/$COMPARISON_MODE"
python -m pytest tests/core/models/test_temporal_hybrid.py -q > "$COMPARISON_ROOT/$COMPARISON_MODE/gpu-correctness.txt" 2>&1
python -m graphids.exp.temporal_comparison "configs/experiments/comparisons/temporal_set_01_${COMPARISON_MODE}_lanes${COMPARISON_LANES}_epoch.yml" --output "$COMPARISON_ROOT/$COMPARISON_MODE"
