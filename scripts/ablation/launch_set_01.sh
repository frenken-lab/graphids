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

# Parent MLflow run per (group, variant) — children link via
# MLFLOW_PARENT_RUN_ID. Keyed "group/variant" → run_id.
declare -A PARENT

_parent_for() {
    # scripts/ablations/<group>/<variant>.jsonnet → PARENT[group/variant]
    local cfg="$1"
    local group variant
    group="$(basename "$(dirname "$cfg")")"
    variant="$(basename "$cfg" .jsonnet)"
    echo "${PARENT[${group}/${variant}]:-}"
}

_open_parents() {
    local pair group variant rid seeds_csv
    seeds_csv="$(IFS=,; echo "${SEEDS[*]}")"
    for pair in "$@"; do
        group=${pair%/*}; variant=${pair#*/}
        if [[ ${#DRY_RUN_FLAG[@]} -gt 0 ]]; then
            PARENT[$pair]="dry_run_$pair"
            echo "  parent $pair → (dry-run)"
            continue
        fi
        if rid=$(python -m graphids mlflow-start-parent \
                --group "$group" --variant "$variant" --dataset "$DATASET" \
                --cluster "$CLUSTER" --seeds "$seeds_csv" 2>/dev/null); then
            PARENT[$pair]="$rid"
            echo "  parent $pair → $rid"
        else
            echo "  parent $pair → (skipped: mlflow unavailable; children un-grouped)"
        fi
    done
}

_submit_fit() {
    # Echoes fit jid on stdout; submission banner flows to stderr.
    # sbatch with --clusters=X prints "Submitted batch job NNN on cluster X" —
    # take field 4 (the jid), not $NF (which is the cluster name).
    local cfg="$1" seed="$2" line
    line=$(MLFLOW_PARENT_RUN_ID="$(_parent_for "$cfg")" \
        scripts/run "$cfg" --dataset "$DATASET" --seed "$seed" --lake-root "$LAKE_ROOT" \
        "${CLUSTER_ARGS[@]}" "${DRY_RUN_FLAG[@]}" 2>&1 | tee /dev/stderr | tail -n 1)
    if [[ ${#DRY_RUN_FLAG[@]} -gt 0 ]]; then echo "0"; else echo "$line" | awk '{print $4}'; fi
}

_chain_test() {
    # afterok CPU test job. Inherits parent MLflow run. Output to stderr so
    # _fit_jid's stdout stays clean for jid capture.
    local cfg="$1" seed="$2" jid="$3"
    local group variant ckpt cmd
    group="$(basename "$(dirname "$cfg")")"
    variant="$(basename "$cfg" .jsonnet)"
    ckpt="${LAKE_ROOT}/${DATASET}/ablations/${group}/${variant}/seed_${seed}/checkpoints/best_model.ckpt"
    cmd="python -m graphids test --config ${cfg} --tla dataset=\"${DATASET}\" --tla seed=${seed} --tla lake_root=\"${LAKE_ROOT}\" --ckpt-path ${ckpt}"
    MLFLOW_PARENT_RUN_ID="$(_parent_for "$cfg")" \
        SBATCH_DEP="afterok:${jid}" \
        scripts/run --mode cpu --command "$cmd" \
        "${CLUSTER_ARGS[@]}" "${DRY_RUN_FLAG[@]}" >&2
}

_fit() {
    # Submit fit + afterok test chain. No stdout capture path.
    local jid
    jid=$(_submit_fit "$@")
    _chain_test "$1" "$2" "$jid"
}
_fit_jid() {
    # Submit fit + afterok test chain; echo fit jid for downstream deps.
    local jid
    jid=$(_submit_fit "$@")
    _chain_test "$1" "$2" "$jid"
    echo "$jid"
}

# -- Open all parent runs upfront (one per group/variant) --------------
echo "=== Opening MLflow parent runs ==="
_open_parents \
    unsupervised/vgae unsupervised/gae unsupervised/dgi \
    conv_type/gat conv_type/gatv2 conv_type/gps \
    gat_sampling/none gat_sampling/curriculum_random gat_sampling/curriculum_vgae \
    gat_loss/ce gat_loss/weighted_ce gat_loss/focal \
    fusion/bandit fusion/dqn fusion/mlp fusion/weighted_avg

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
declare -A STATES_JID
echo "=== Stage 3: extract-fusion-states (${#SEEDS[@]} jobs) ==="
for SEED in "${SEEDS[@]}"; do
    VGAE_CKPT="${LAKE_ROOT}/${DATASET}/ablations/unsupervised/vgae/seed_${SEED}/checkpoints/best_model.ckpt"
    GAT_CKPT="${LAKE_ROOT}/${DATASET}/ablations/gat_loss/focal/seed_${SEED}/checkpoints/best_model.ckpt"
    OUT="${LAKE_ROOT}/${DATASET}/ablations/fusion_states/seed_${SEED}"
    CMD="python -m graphids extract-fusion-states \
--vgae-ckpt ${VGAE_CKPT} --gat-ckpt ${GAT_CKPT} \
--dataset ${DATASET} --seed ${SEED} --output-dir ${OUT}"
    line=$(SBATCH_DEP="afterok:${VGAE_JID[$SEED]}:${FOCAL_JID[$SEED]}" \
        scripts/run --mode gpu --mem 36G --time 0:30:00 --command "$CMD" \
        "${DRY_RUN_FLAG[@]}" 2>&1 | tee /dev/stderr | tail -n 1)
    # Extract jid from "Submitted batch job NNN on cluster X" (field 4).
    if [[ ${#DRY_RUN_FLAG[@]} -gt 0 ]]; then
        STATES_JID[$SEED]="0"
    else
        STATES_JID[$SEED]=$(echo "$line" | awk '{print $4}')
    fi
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
