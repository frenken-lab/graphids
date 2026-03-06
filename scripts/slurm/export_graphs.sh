#!/usr/bin/env bash
# Export graph samples + statistics to reports/data/.
# CPU-only, no GPU needed. Submit with:
#   bash scripts/slurm/export_graphs.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$PROJECT_DIR/slurm_logs"

set -a; source "$PROJECT_DIR/.env" 2>/dev/null; set +a

sbatch --account="${KD_GAT_SLURM_ACCOUNT:?Set KD_GAT_SLURM_ACCOUNT in .env}" --partition=cpu \
  --time=60 --mem=48G --cpus-per-task=4 \
  --job-name=graph-export --output="$PROJECT_DIR/slurm_logs/%j-graph-export.out" \
  --error="$PROJECT_DIR/slurm_logs/%j-graph-export.err" \
  --wrap="cd $PROJECT_DIR && module load python/3.12 && source .venv/bin/activate && source $PROJECT_DIR/.env && python -m graphids.pipeline.export --graphs --output-dir reports/data && echo DONE"

echo "Submitted graph export job. Check slurm_logs/ for output."
