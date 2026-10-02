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
#SBATCH --job-name=temporal-lane-sweep

set -euo pipefail
cd "${SWEEP_CODE_ROOT:-/users/PAS2022/rf15/projects/lab/graphids}"
source scripts/slurm/_preamble.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SWEEP_ROOT:?set SWEEP_ROOT to a unique result directory}"
mkdir -p "$SWEEP_ROOT"
python -c 'import torch; assert torch.cuda.is_available(), "GPU sweep requires CUDA"'
python -m pytest tests/core/data tests/core/models/test_temporal_hybrid.py tests/exp/test_temporal_lane_sweep.py tests/exp/test_config_representation.py -q > "$SWEEP_ROOT/gpu-correctness.txt" 2>&1
python -m graphids.exp.temporal_lane_sweep "$SWEEP_ROOT" --note /users/PAS2022/rf15/projects/lab/graphids/docs/empirical-notes/2026-09-29-temporal-lane-throughput-check.md
python scripts/lint_docs.py
