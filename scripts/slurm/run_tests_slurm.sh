#!/usr/bin/env bash
# Submit pytest to a SLURM compute node (cpu partition, no GPU needed).
#
# Usage:
#   bash scripts/slurm/run_tests_slurm.sh                       # all tests
#   bash scripts/slurm/run_tests_slurm.sh -k "test_smoke"       # specific test
#   bash scripts/slurm/run_tests_slurm.sh -m "not slow"         # skip slow tests
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$PROJECT_DIR/slurm_logs"

# Source .env for KD_GAT_SLURM_ACCOUNT
set -a; source "$PROJECT_DIR/.env" 2>/dev/null; set +a

sbatch --account="${KD_GAT_SLURM_ACCOUNT:?Set KD_GAT_SLURM_ACCOUNT in .env}" --partition=cpu \
  --time=00:30:00 --mem=16G --cpus-per-task=4 \
  --job-name=kd-gat-pytest --output="$PROJECT_DIR/slurm_logs/%j-pytest.out" \
  --error="$PROJECT_DIR/slurm_logs/%j-pytest.err" \
  --wrap="SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 source $PROJECT_DIR/scripts/slurm/_preamble.sh && python -m pytest tests/ -v $*"

echo "Submitted pytest job. Check slurm_logs/ for output."
