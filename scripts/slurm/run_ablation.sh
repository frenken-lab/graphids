#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --account=PAS1266
#SBATCH --job-name=dagster-ablation
#SBATCH --output=slurm_logs/dagster_%j.out
#SBATCH --error=slurm_logs/dagster_%j.err
#SBATCH --signal=B:USR1@300
set -euo pipefail

# --- CPU orchestrator job ---
# Dagster materializes all assets per partition (dataset|seed).
# Each asset submits its own GPU sbatch job, polls sacct, handles retry.
# Restart-safe: re-running skips completed stages (best_model.ckpt check).
#
# Usage:
#   sbatch scripts/slurm/run_ablation.sh                        # ablation (default)
#   sbatch scripts/slurm/run_ablation.sh --recipe production.yaml
#   sbatch scripts/slurm/run_ablation.sh --dataset set_01       # single dataset

SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 \
    source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh"

RECIPE="${KD_GAT_RECIPE:-ablation.yaml}"
DATASETS=""
SEEDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recipe) RECIPE="$2"; shift 2 ;;
        --dataset) DATASETS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Read defaults from recipe if not overridden
RECIPE_PATH="graphids/config/recipes/${RECIPE}"
if [[ ! -f "${RECIPE_PATH}" ]]; then
    echo "Recipe not found: ${RECIPE_PATH}" >&2
    exit 1
fi

if [[ -z "${DATASETS}" ]]; then
    DATASETS=$(python -c "import yaml; r=yaml.safe_load(open('${RECIPE_PATH}')); print(' '.join(r['sweep']['datasets']))")
fi
if [[ -z "${SEEDS}" ]]; then
    SEEDS=$(python -c "import yaml; r=yaml.safe_load(open('${RECIPE_PATH}')); print(' '.join(str(s) for s in r['sweep']['seeds']))")
fi

echo "=== Pipeline Run ==="
echo "Recipe:   ${RECIPE}"
echo "Datasets: ${DATASETS}"
echo "Seeds:    ${SEEDS}"
echo "====================="

for dataset in ${DATASETS}; do
    for seed in ${SEEDS}; do
        echo ""
        echo ">>> Materializing: ${dataset}|${seed}"
        python -m graphids.orchestrate run \
            --recipe "${RECIPE_PATH}" \
            --dataset "${dataset}" \
            --seed "${seed}"
        echo "<<< Done: ${dataset}|${seed}"
    done
done

echo ""
echo "=== All partitions complete ==="
