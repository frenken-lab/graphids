#!/usr/bin/env bash
# Account comes from .env (KD_GAT_SLURM_ACCOUNT). Submit with:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/profiling/benchmark_orchestration.sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=06:00:00
#SBATCH --job-name=bench-orch
#SBATCH --output=slurm_logs/bench_orch_%j.out
#SBATCH --error=slurm_logs/bench_orch_%j.err

# Orchestration benchmark: measure subprocess dispatch overhead, inter-stage
# GPU idle gaps, and per-stage wall times.
#
# Usage:
#   sbatch scripts/profiling/benchmark_orchestration.sh
#   sbatch scripts/profiling/benchmark_orchestration.sh hcrl_ch   # smaller dataset
#
# Produces benchmark_timing.jsonl with per-stage JSONL records.
# Phase R2 analysis is pending R1 benchmark data from this script.

set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

# --- Environment ---
module load python/3.12
source .venv/bin/activate

# Source project env vars
set -a
source .env
set +a

# Stage data to fast storage
source scripts/data/stage_data.sh --cache

# --- Benchmark config ---
DATASET="${1:-hcrl_ch}"
BENCHMARK_LOG="slurm_logs/benchmark_timing_${SLURM_JOB_ID}.jsonl"

echo "=== Orchestration Benchmark ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Dataset:   ${DATASET}"
echo "Log:       ${BENCHMARK_LOG}"
echo "Python:    $(which python)"
echo "PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Pre-benchmark GPU baseline
echo "=== Pre-benchmark GPU state ==="
nvidia-smi 2>/dev/null || true
echo ""

# Enable benchmark instrumentation
export KD_GAT_BENCHMARK=1
export KD_GAT_BENCHMARK_LOG="${BENCHMARK_LOG}"

# Run the full pipeline for one dataset (all variants: large, small_kd, small_nokd)
PIPELINE_START=$(date +%s)

python -m graphids.pipeline.cli flow --dataset "${DATASET}" --local

PIPELINE_END=$(date +%s)
PIPELINE_ELAPSED=$((PIPELINE_END - PIPELINE_START))

echo ""
echo "=== Benchmark Results ==="
echo "Total pipeline wall time: ${PIPELINE_ELAPSED}s"
echo "Timing log: ${BENCHMARK_LOG}"
echo ""

# Print summary from JSONL
if [[ -f "${BENCHMARK_LOG}" ]]; then
    echo "=== Per-stage timing summary ==="
    python -c "
import json, sys

records = []
with open('${BENCHMARK_LOG}') as f:
    for line in f:
        records.append(json.loads(line))

print(f'{'Stage':<40} {'Spawn':>8} {'Exec':>10} {'Total':>10} {'Gap':>8}')
print('-' * 80)
total_spawn = 0.0
total_gap = 0.0
for r in records:
    label = f\"{r['model']}/{r['scale']}/{r['stage']}\"
    if r['auxiliaries'] != 'none':
        label += f\" ({r['auxiliaries']})\"
    spawn = r['spawn_overhead_s']
    total_spawn += spawn
    gap = r.get('inter_stage_gap_s') or 0.0
    total_gap += gap
    print(f\"{label:<40} {spawn:>7.3f}s {r['execution_s']:>9.1f}s {r['total_s']:>9.1f}s {gap:>7.3f}s\")

print('-' * 80)
print(f'Total spawn overhead: {total_spawn:.3f}s')
print(f'Total inter-stage gaps: {total_gap:.3f}s')
print(f'Stages: {len(records)}')
"
fi

# --- Post-job ---
source scripts/slurm/job_epilog.sh
