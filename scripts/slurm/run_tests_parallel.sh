#!/usr/bin/env bash
# Submit each test file as a separate SLURM job for parallel execution.
# Usage: bash scripts/slurm/run_tests_parallel.sh [extra pytest args]
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$PROJECT_DIR/slurm_logs"

# Source .env for KD_GAT_SLURM_ACCOUNT
set -a; source "$PROJECT_DIR/.env" 2>/dev/null; set +a

for f in "$PROJECT_DIR"/tests/test_*.py; do
    name=$(basename "$f" .py)
    sbatch --account="${KD_GAT_SLURM_ACCOUNT:?Set KD_GAT_SLURM_ACCOUNT in .env}" --partition=cpu \
      --time=30 --mem=32G --cpus-per-task=4 \
      --job-name="$name" \
      --output="$PROJECT_DIR/slurm_logs/%j-$name.out" \
      --error="$PROJECT_DIR/slurm_logs/%j-$name.err" \
      --wrap="cd $PROJECT_DIR && module load python/3.12 && source .venv/bin/activate && python -m pytest $f -v --run-slurm $*"
    echo "Submitted: $name"
done
echo "All test jobs submitted. Check: squeue -u \$USER"
