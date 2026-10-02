#!/usr/bin/env bash
#SBATCH --clusters=pitzer
#SBATCH --account=pas1266
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=temporal-cuda-graph
set -euo pipefail
cd "${GRAPH_CODE_ROOT:-/users/PAS2022/rf15/projects/lab/graphids}"
source scripts/slurm/_diagnostic.sh
: "${GRAPH_ROOT:?set GRAPH_ROOT to a unique shared output directory}"
diagnostic_init "$GRAPH_ROOT" V100
python -m pytest tests/core/models/test_temporal_cuda_graph.py tests/core/models/test_temporal_hybrid.py -q -k "${GRAPH_TEST_FILTER:-}" > "$GRAPH_ROOT/correctness.txt" 2>&1
if [[ "${GRAPH_CORRECTNESS_ONLY:-0}" == "1" ]]; then exit 0; fi
python -m graphids.exp.temporal_cuda_graph "$GRAPH_ROOT"
