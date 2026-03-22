#!/usr/bin/env bash
#SBATCH --partition=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=kd-gat-pipeline-test
#SBATCH --output=slurm_logs/pipeline_test_%j.out
#SBATCH --error=slurm_logs/pipeline_test_%j.err
#SBATCH --signal=B:USR1@300

# Pipeline variation test: runs each stage sequentially, verifying
# inter-stage checkpoints exist before proceeding.
# Usage: sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/test_pipeline_stages.sbatch

source /users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh

DATASET="hcrl_ch"
SCALE="small"
SEED="42"
LAKE_ROOT="${KD_GAT_LAKE_ROOT:-experimentruns}"
STAGE_BASE="${LAKE_ROOT}/dev/${USER}/${DATASET}"
PASSED=0
FAILED=0

run_stage() {
    local desc="$1"; shift
    echo "=== $desc ==="
    if python -m graphids "$@"; then
        echo "PASS: $desc"
        ((PASSED++))
    else
        echo "FAIL: $desc (exit $?)"
        ((FAILED++))
        return 1
    fi
}

check_checkpoint() {
    local desc="$1"
    local ckpt="$2"
    if [[ -f "$ckpt" ]]; then
        echo "  checkpoint verified: $ckpt"
    else
        echo "  MISSING checkpoint: $ckpt"
        echo "  $desc cannot proceed — aborting pipeline"
        ((FAILED++))
        return 1
    fi
}

log_job_header "Pipeline stage tests: $DATASET $SCALE"
echo "LAKE_ROOT: $LAKE_ROOT"
echo "STAGE_BASE: $STAGE_BASE"
echo ""

# Stage 1: VGAE autoencoder
VGAE_CKPT="${STAGE_BASE}/vgae_${SCALE}_autoencoder/seed_${SEED}/best_model.pt"
run_stage "VGAE autoencoder" stage=autoencoder model_type=vgae scale=$SCALE dataset=$DATASET \
    training.max_epochs=2 training.patience=5 seed=$SEED || { echo "Pipeline aborted at stage 1"; exit 1; }
check_checkpoint "GAT curriculum" "$VGAE_CKPT" || { echo "Pipeline aborted: VGAE checkpoint missing"; exit 1; }

# Stage 2: GAT curriculum (depends on VGAE checkpoint)
GAT_CKPT="${STAGE_BASE}/gat_${SCALE}_curriculum/seed_${SEED}/best_model.pt"
run_stage "GAT curriculum" stage=curriculum model_type=gat scale=$SCALE dataset=$DATASET \
    training.max_epochs=2 training.patience=5 seed=$SEED || { echo "Pipeline aborted at stage 2"; exit 1; }
check_checkpoint "DQN fusion" "$GAT_CKPT" || { echo "Pipeline aborted: GAT checkpoint missing"; exit 1; }

# Stage 3: DQN fusion (depends on VGAE + GAT checkpoints)
DQN_CKPT="${STAGE_BASE}/dqn_${SCALE}_fusion/seed_${SEED}/best_model.pt"
run_stage "DQN fusion" stage=fusion model_type=dqn scale=$SCALE dataset=$DATASET \
    fusion.episodes=5 seed=$SEED || { echo "Pipeline aborted at stage 3"; exit 1; }
check_checkpoint "Evaluation" "$DQN_CKPT" || { echo "Pipeline aborted: DQN checkpoint missing"; exit 1; }

# Stage 4: Evaluation (depends on all 3 checkpoints)
run_stage "Evaluation" stage=evaluation model_type=vgae scale=$SCALE dataset=$DATASET seed=$SEED

echo ""
echo "========================================="
echo "Results: $PASSED passed, $FAILED failed"
echo "========================================="

JOB_LOG_PREFIX="pipeline_test" source /users/PAS2022/rf15/KD-GAT/scripts/slurm/_epilog.sh
exit $FAILED
