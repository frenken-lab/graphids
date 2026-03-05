#!/bin/bash
# Build test graph caches — submits one SLURM job per dataset.
# Usage: bash scripts/data/build_test_cache.sh                    # All datasets (set_01-04)
#        bash scripts/data/build_test_cache.sh set_02 set_03      # Specific datasets

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$PROJECT_DIR/slurm_logs"

# Source .env for KD_GAT_SLURM_ACCOUNT
set -a; source "$PROJECT_DIR/.env" 2>/dev/null; set +a

if [ $# -gt 0 ]; then
    DATASETS="$@"
else
    DATASETS="set_01 set_02 set_03 set_04"
fi

for ds in $DATASETS; do
    sbatch --account="${KD_GAT_SLURM_ACCOUNT:?Set KD_GAT_SLURM_ACCOUNT in .env}" --partition=cpu \
      --time=240 --mem=85G --cpus-per-task=8 \
      --job-name="test-cache-${ds}" \
      --output="$PROJECT_DIR/slurm_logs/%j-test-cache-${ds}.out" \
      --error="$PROJECT_DIR/slurm_logs/%j-test-cache-${ds}.err" \
      --wrap="source $PROJECT_DIR/.venv/bin/activate && cd $PROJECT_DIR && python -c \"
from graphids.core.training.datamodules import load_test_scenarios
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
ds = '${ds}'
print(f'=== Building test cache for {ds} ===', flush=True)
scenarios = load_test_scenarios(ds, Path(f'data/automotive/{ds}'), Path(f'data/cache/{ds}'))
for name, graphs in scenarios.items():
    print(f'  {name}: {len(graphs)} graphs', flush=True)
print(f'=== Done: {ds} ({len(scenarios)} scenarios) ===', flush=True)
\""
    echo "Submitted test-cache job for ${ds}"
done
