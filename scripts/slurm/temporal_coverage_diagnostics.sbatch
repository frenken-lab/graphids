#!/usr/bin/env bash
#SBATCH --clusters=pitzer
#SBATCH --account=pas1266
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=temporal-coverage-diagnostics

set -euo pipefail
cd /users/PAS2022/rf15/projects/lab/graphids
source scripts/slurm/_preamble.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${COMPARISON_ROOT:?set COMPARISON_ROOT to a unique result directory}"
mkdir -p "$COMPARISON_ROOT"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA correctness must run on a GPU"'
python -m pytest tests/core/data tests/core/models tests/exp -q > "$COMPARISON_ROOT/gpu-correctness.txt" 2>&1
python -m graphids.exp.temporal_comparison configs/experiments/comparisons/temporal_set_01_segmented_lanes32_diagnostic.yml --output "$COMPARISON_ROOT/diagnostic32"
python -m graphids.exp.temporal_comparison configs/experiments/comparisons/temporal_set_01_segmented_lanes64_diagnostic.yml --output "$COMPARISON_ROOT/diagnostic64"
