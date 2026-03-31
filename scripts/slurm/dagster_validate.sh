#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --account=PAS1266
#SBATCH --job-name=dagster-validate
#SBATCH --output=slurm_logs/dagster-validate_%j.out
#SBATCH --error=slurm_logs/dagster-validate_%j.err

SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh"

python -m graphids.orchestrate validate
