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
source scripts/slurm/_diagnostic.sh
: "${SWEEP_ROOT:?set SWEEP_ROOT to a unique result directory}"
diagnostic_init "$SWEEP_ROOT"
python -m pytest tests/core/data tests/core/models/test_temporal_hybrid.py tests/exp/test_temporal_lane_sweep.py tests/exp/test_config_representation.py -q > "$SWEEP_ROOT/gpu-correctness.txt" 2>&1
python -m graphids.exp.temporal_lane_sweep "$SWEEP_ROOT" --note /users/PAS2022/rf15/projects/lab/graphids/docs/empirical-notes/2026-09-29-temporal-lane-throughput-check.md
python scripts/lint_docs.py
