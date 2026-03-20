#!/usr/bin/env bash
#SBATCH --account=PAS1266
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=kd-gat-migrate-ess
#SBATCH --output=slurm_logs/migrate_ess_%j.out
#SBATCH --error=slurm_logs/migrate_ess_%j.err

# No GPU, no data staging needed for migration
SKIP_CUDA_CONF=1
SKIP_STAGE_DATA=1
source scripts/slurm/_preamble.sh

log_job_header "ESS Data Lake Migration"

bash scripts/lake/migrate_to_ess.sh 2>&1

echo ""
echo "=== Building DuckDB catalog ==="
python -m graphids.lake --action rebuild-catalog

echo ""
echo "=== Verifying artifacts ==="
python -m graphids.lake --action status

EXIT_CODE=$?
log_job_footer $EXIT_CODE
exit $EXIT_CODE
