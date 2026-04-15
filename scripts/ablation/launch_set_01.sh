#!/usr/bin/env bash
# Launch the set_01 ablation study with SLURM dependency chains.
# See docs/plans/ablation-set_01.md for the runbook.
#
# Delegates per-job submission to ``scripts/run``, which owns TLA
# construction + resource lookup. This script only owns the DAG
# shape (which presets, which seeds, which afterok edges).
#
# Usage:
#   scripts/ablation/launch_set_01.sh                # submit everything
#   scripts/ablation/launch_set_01.sh --dry-run      # print commands, no submit
#   scripts/ablation/launch_set_01.sh --seed 42      # only one seed
#   scripts/ablation/launch_set_01.sh --cluster cardinal  # target cluster
set -euo pipefail

DRY_RUN_FLAG=()
SEEDS=(42 123 777)
CLUSTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN_FLAG=(--dry-run); shift ;;
        --seed)     SEEDS=("$2"); shift 2 ;;
        --cluster)  CLUSTER="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
source .env

DATASET=set_01
LAKE_ROOT="${GRAPHIDS_LAKE_ROOT:?GRAPHIDS_LAKE_ROOT must be set in .env}/dev/${USER}"

CLUSTER_ARGS=()
[[ -n "$CLUSTER" ]] && CLUSTER_ARGS=(--cluster "$CLUSTER")

_fit() {
    local cfg="$1" seed="$2"
    scripts/run "$cfg" --dataset "$DATASET" --seed "$seed" --lake-root "$LAKE_ROOT" \
        "${CLUSTER_ARGS[@]}" "${DRY_RUN_FLAG[@]}"
}
_fit_jid() {
    local line
    line=$(_fit "$@" 2>&1 | tee /dev/stderr | tail -n 1)
    [[ ${#DRY_RUN_FLAG[@]} -gt 0 ]] && echo "0" || echo "${line##* }"
}

# -- Stage 0: baseline VGAEs -------------------------------------------
declare -A VGAE_JID
echo "=== Stage 0: baseline VGAEs (${#SEEDS[@]} jobs) ==="
for SEED in "${SEEDS[@]}"; do
    VGAE_JID[$SEED]=$(_fit_jid configs/ablations/unsupervised/vgae.jsonnet "$SEED")
    echo "  seed=${SEED} -> vgae jid=${VGAE_JID[$SEED]}"
done

# -- Stage 1: standalone ablation groups -------------------------------
declare -A FOCAL_JID
echo "=== Stage 1: standalone (10 × ${#SEEDS[@]} jobs) ==="
for SEED in "${SEEDS[@]}"; do
    for CT in gat gatv2 gps; do
        _fit "configs/ablations/conv_type/${CT}.jsonnet" "$SEED"
    done
    for VAR in gae dgi; do
        _fit "configs/ablations/unsupervised/${VAR}.jsonnet" "$SEED"
    done
    for SMP in none curriculum_random; do
        _fit "configs/ablations/gat_sampling/${SMP}.jsonnet" "$SEED"
    done
    for LOSS in ce weighted_ce; do
        _fit "configs/ablations/gat_loss/${LOSS}.jsonnet" "$SEED"
    done
    # focal doubles as the baseline GAT for fusion — capture its jid.
    FOCAL_JID[$SEED]=$(_fit_jid configs/ablations/gat_loss/focal.jsonnet "$SEED")
    echo "  seed=${SEED} -> focal jid=${FOCAL_JID[$SEED]}"
done

# -- Stage 2: curriculum_vgae (afterok Stage 0) ------------------------
echo "=== Stage 2: curriculum_vgae (${#SEEDS[@]} jobs, afterok Stage 0) ==="
for SEED in "${SEEDS[@]}"; do
    SBATCH_DEP="afterok:${VGAE_JID[$SEED]}" \
        _fit configs/ablations/gat_sampling/curriculum_vgae.jsonnet "$SEED"
done

# -- Stage 3: extract-fusion-states (afterok Stage 0 + Stage 1 focal) --
# extract-fusion-states isn't an ablation preset — it's an op, keep using submit.sh.
declare -A STATES_JID
echo "=== Stage 3: extract-fusion-states (${#SEEDS[@]} jobs) ==="
for SEED in "${SEEDS[@]}"; do
    VGAE_CKPT="${LAKE_ROOT}/${DATASET}/ablations/unsupervised/vgae/seed_${SEED}/best.ckpt"
    GAT_CKPT="${LAKE_ROOT}/${DATASET}/ablations/gat_loss/focal/seed_${SEED}/best.ckpt"
    OUT="${LAKE_ROOT}/${DATASET}/ablations/fusion_states/seed_${SEED}"
    line=$(SBATCH_DEP="afterok:${VGAE_JID[$SEED]}:${FOCAL_JID[$SEED]}" \
        scripts/slurm/submit.sh extract-fusion-states \
        --vgae-ckpt "${VGAE_CKPT}" --gat-ckpt "${GAT_CKPT}" \
        --dataset "${DATASET}" --seed "${SEED}" --output-dir "${OUT}" \
        2>&1 | tee /dev/stderr | tail -n 1)
    STATES_JID[$SEED]="${line##* }"
    echo "  seed=${SEED} -> states jid=${STATES_JID[$SEED]}"
done

# -- Stage 4: fusion ablations (afterok Stage 3) -----------------------
echo "=== Stage 4: fusion ablations (4 × ${#SEEDS[@]} jobs) ==="
for SEED in "${SEEDS[@]}"; do
    for METHOD in bandit dqn mlp weighted_avg; do
        SBATCH_DEP="afterok:${STATES_JID[$SEED]}" \
            _fit "configs/ablations/fusion/${METHOD}.jsonnet" "$SEED"
    done
done

echo ""
echo "=== Launched ==="
echo "Stage 0 VGAE jids:  ${VGAE_JID[*]}"
echo "Stage 1 focal jids: ${FOCAL_JID[*]}"
echo "Stage 3 state jids: ${STATES_JID[*]}"
echo "Monitor: squeue -u \$USER -o '%.10i %.10P %.20j %.2t %.10M %R' | head -60"
