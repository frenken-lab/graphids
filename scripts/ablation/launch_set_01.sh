#!/usr/bin/env bash
# Launch the set_01 ablation study with SLURM dependency chains.
# See docs/plans/ablation-set_01.md for the runbook.
#
# Each ablation jsonnet auto-computes its run_dir and upstream ckpt paths
# from (dataset, seed) via configs/ablations/_paths.libsonnet, so submit
# calls only pass --tla dataset + --tla seed.
#
# Usage:
#   scripts/ablation/launch_set_01.sh                # submit everything
#   scripts/ablation/launch_set_01.sh --dry-run      # print commands, no submit
#   scripts/ablation/launch_set_01.sh --seed 42      # only one seed
set -euo pipefail

DRY_RUN=""
SEEDS=(42 123 777)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="echo DRY:"; shift ;;
        --seed) SEEDS=("$2"); shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
source .env  # pulls GRAPHIDS_LAKE_ROOT, account, etc.

DATASET=set_01
LAKE_ROOT="${GRAPHIDS_LAKE_ROOT:?GRAPHIDS_LAKE_ROOT must be set in .env}/dev/${USER}"
TLA_LAKE="--tla lake_root=\"${LAKE_ROOT}\""

_capture_jid() {
    local line
    line=$($DRY_RUN "$@" 2>&1 | tee /dev/stderr | tail -n 1)
    [[ -n "$DRY_RUN" ]] && echo "0" || echo "${line##* }"
}

# Helper: submit via `fit-long` profile with dataset/seed/lake_root TLAs.
_submit_fit() {
    local cfg="$1" seed="$2"
    $DRY_RUN scripts/slurm/submit.sh fit-long --config "$cfg" \
        --tla "dataset=\"${DATASET}\"" --tla "seed=${seed}" \
        --tla "lake_root=\"${LAKE_ROOT}\""
}
_capture_fit() {
    local cfg="$1" seed="$2"
    _capture_jid scripts/slurm/submit.sh fit-long --config "$cfg" \
        --tla "dataset=\"${DATASET}\"" --tla "seed=${seed}" \
        --tla "lake_root=\"${LAKE_ROOT}\""
}

# -- Stage 0: baseline VGAEs -------------------------------------------
declare -A VGAE_JID
echo "=== Stage 0: baseline VGAEs (${#SEEDS[@]} jobs) ==="
for SEED in "${SEEDS[@]}"; do
    VGAE_JID[$SEED]=$(_capture_fit configs/ablations/unsupervised/vgae.jsonnet "$SEED")
    echo "  seed=${SEED} -> vgae jid=${VGAE_JID[$SEED]}"
done

# -- Stage 1: standalone ablation groups -------------------------------
declare -A FOCAL_JID
echo "=== Stage 1: standalone (10 × ${#SEEDS[@]} jobs) ==="
for SEED in "${SEEDS[@]}"; do
    for CT in gat gatv2 gps; do
        _submit_fit "configs/ablations/conv_type/${CT}.jsonnet" "$SEED"
    done
    for VAR in gae dgi; do
        _submit_fit "configs/ablations/unsupervised/${VAR}.jsonnet" "$SEED"
    done
    for SMP in none curriculum_random; do
        _submit_fit "configs/ablations/gat_sampling/${SMP}.jsonnet" "$SEED"
    done
    for LOSS in ce weighted_ce; do
        _submit_fit "configs/ablations/gat_loss/${LOSS}.jsonnet" "$SEED"
    done
    # focal is captured for Stage 3 (doubles as the baseline GAT for fusion).
    FOCAL_JID[$SEED]=$(_capture_fit configs/ablations/gat_loss/focal.jsonnet "$SEED")
    echo "  seed=${SEED} -> focal jid=${FOCAL_JID[$SEED]}"
done

# -- Stage 2: curriculum_vgae (afterok Stage 0) ------------------------
echo "=== Stage 2: curriculum_vgae (${#SEEDS[@]} jobs, afterok Stage 0) ==="
for SEED in "${SEEDS[@]}"; do
    SBATCH_DEP="afterok:${VGAE_JID[$SEED]}" \
        _submit_fit configs/ablations/gat_sampling/curriculum_vgae.jsonnet "$SEED"
done

# -- Stage 3: extract-fusion-states (afterok Stage 0 + Stage 1 focal) --
declare -A STATES_JID
echo "=== Stage 3: extract-fusion-states (${#SEEDS[@]} jobs) ==="
for SEED in "${SEEDS[@]}"; do
    VGAE_CKPT="${LAKE_ROOT}/${DATASET}/ablations/unsupervised/vgae/seed_${SEED}/best.ckpt"
    GAT_CKPT="${LAKE_ROOT}/${DATASET}/ablations/gat_loss/focal/seed_${SEED}/best.ckpt"
    OUT="${LAKE_ROOT}/${DATASET}/ablations/fusion_states/seed_${SEED}"
    STATES_JID[$SEED]=$(SBATCH_DEP="afterok:${VGAE_JID[$SEED]}:${FOCAL_JID[$SEED]}" \
        _capture_jid scripts/slurm/submit.sh extract-fusion-states \
        --vgae-ckpt "${VGAE_CKPT}" --gat-ckpt "${GAT_CKPT}" \
        --dataset "${DATASET}" --seed "${SEED}" --output-dir "${OUT}")
    echo "  seed=${SEED} -> states jid=${STATES_JID[$SEED]}"
done

# -- Stage 4: fusion ablations (afterok Stage 3) -----------------------
echo "=== Stage 4: fusion ablations (4 × ${#SEEDS[@]} jobs) ==="
for SEED in "${SEEDS[@]}"; do
    for METHOD in bandit dqn mlp weighted_avg; do
        SBATCH_DEP="afterok:${STATES_JID[$SEED]}" \
            _submit_fit "configs/ablations/fusion/${METHOD}.jsonnet" "$SEED"
    done
done

echo ""
echo "=== Launched ==="
echo "Stage 0 VGAE jids:  ${VGAE_JID[*]}"
echo "Stage 1 focal jids: ${FOCAL_JID[*]}"
echo "Stage 3 state jids: ${STATES_JID[*]}"
echo "Monitor: squeue -u \$USER -o '%.10i %.10P %.20j %.2t %.10M %R' | head -60"
