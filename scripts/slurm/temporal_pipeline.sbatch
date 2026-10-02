#!/usr/bin/env bash
#SBATCH --clusters=pitzer
#SBATCH --account=pas1266
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=temporal-pipeline-256
set -euo pipefail
cd "${PIPELINE_CODE_ROOT:-/users/PAS2022/rf15/projects/lab/graphids}"
source scripts/slurm/_diagnostic.sh
: "${PIPELINE_ROOT:?set PIPELINE_ROOT}"
diagnostic_init "$PIPELINE_ROOT"
python -m pytest tests/core/data tests/core/models/test_temporal_hybrid.py tests/core/models/test_temporal_ssm_kernel.py -q > "$PIPELINE_ROOT/correctness.txt" 2>&1
args=()
if [[ "${PIPELINE_FUSION:-0}" == 1 ]]; then args+=(--fusion); fi
python -m graphids.exp.temporal_pipeline_ab "$PIPELINE_ROOT" "${args[@]}"
