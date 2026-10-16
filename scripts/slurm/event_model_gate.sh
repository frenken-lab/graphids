#!/usr/bin/env bash
#SBATCH --clusters=ascend
#SBATCH --account=pas1266
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-consolidation-2026-10-16/gpu.log
set -euo pipefail
cd /users/PAS2022/rf15/projects/lab/graphids
module load python/3.12 cuda/12.8.1
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/event_model_gate.py --mode gpu --output /fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-consolidation-2026-10-16/gpu
