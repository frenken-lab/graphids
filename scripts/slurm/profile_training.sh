#!/bin/bash
#SBATCH --job-name=kd-gat-profile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --account=PAS1266
#SBATCH --signal=B:USR1@60
#SBATCH --output=slurm_logs/profile_%j.out
#SBATCH --error=slurm_logs/profile_%j.err

# Repeatable profiling tool for DataLoader bottleneck + GPU utilization.
#
# Default: VGAE on hcrl_ch, 2 workers, 5 epochs, SimpleProfiler.
# Override via env vars:
#
#   PROFILE_DATASET=set_02 PROFILE_WORKERS=3 PROFILE_MODEL=gat sbatch scripts/slurm/profile_training.sh
#   PROFILE_PROFILER=advanced sbatch scripts/slurm/profile_training.sh
#   PROFILE_PROFILER=pytorch sbatch scripts/slurm/profile_training.sh  # chrome trace

module load python/3.12
source /users/PAS2022/rf15/KD-GAT/.venv/bin/activate
source /users/PAS2022/rf15/KD-GAT/.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET="${PROFILE_DATASET:-hcrl_ch}"
WORKERS="${PROFILE_WORKERS:-2}"
EPOCHS="${PROFILE_EPOCHS:-5}"
PROFILER="${PROFILE_PROFILER:-simple}"
MODEL="${PROFILE_MODEL:-vgae}"

# Map model to class path
case "$MODEL" in
    vgae) MODEL_CLASS="graphids.core.models.vgae.VGAEModule" ;;
    gat)  MODEL_CLASS="graphids.core.models.gat.GATModule" ;;
    dgi)  MODEL_CLASS="graphids.core.models.dgi.DGIModule" ;;
    *)    echo "Unknown model: $MODEL"; exit 1 ;;
esac

# Map profiler name to CLI args
case "$PROFILER" in
    simple)   PROFILER_ARGS="--trainer.profiler=simple" ;;
    advanced) PROFILER_ARGS="--trainer.profiler=advanced" ;;
    pytorch)  PROFILER_ARGS="--trainer.profiler=pytorch_lightning.profilers.PyTorchProfiler --trainer.profiler.filename=profile_${MODEL}_${DATASET} --trainer.profiler.profile_memory=true --trainer.profiler.row_limit=30" ;;
    *)        echo "Unknown profiler: $PROFILER"; exit 1 ;;
esac

echo "=== Profile Run ==="
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Model:    $MODEL ($MODEL_CLASS)"
echo "Dataset:  $DATASET"
echo "Workers:  $WORKERS"
echo "Epochs:   $EPOCHS"
echo "Profiler: $PROFILER"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "==================="

srun python -m graphids fit \
    --config configs/profile.yaml \
    --model "$MODEL_CLASS" \
    --data.dataset="$DATASET" \
    --data.num_workers="$WORKERS" \
    --trainer.max_epochs="$EPOCHS" \
    $PROFILER_ARGS

EXIT=$?
echo "=== Exit: $EXIT ==="

# Memory summary from SLURM
sacct -j "$SLURM_JOB_ID" --format=JobID%15,Elapsed,MaxRSS%12,MaxVMSize%12 --noheader 2>/dev/null
