#!/usr/bin/env bash
#SBATCH --clusters=pitzer
#SBATCH --account=pas1266
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=temporal-coverage-report

set -euo pipefail
cd /users/PAS2022/rf15/projects/lab/graphids
export SKIP_CUDA_CONF=1
source scripts/slurm/_preamble.sh
: "${COMPARISON_ROOT:?set COMPARISON_ROOT}"
python -m graphids.exp.temporal_results "$COMPARISON_ROOT" --note docs/empirical-notes/2026-09-29-temporal-lane-throughput-check.md
python scripts/lint_docs.py
