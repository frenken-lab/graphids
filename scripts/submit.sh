#!/usr/bin/env bash
# Unified SLURM job launcher. Run from login node.
# Resource profiles are read from graphids/config/resources.yaml via Python.
#
# Usage:
#   scripts/submit.sh tests [-k pattern] [-x]
#   scripts/submit.sh rebuild-caches [--dataset hcrl_ch | --all] [--delete-existing]
#   scripts/submit.sh validate
#   scripts/submit.sh landscape <model_type> <dataset> <ckpt_path> [--resolution N]
#   scripts/submit.sh ablation [--recipe X --dataset X --seed X]
#   scripts/submit.sh profile [stage scale dataset]
#   scripts/submit.sh probe-budget [args]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .env
SLURM_LOG_DIR="${KD_GAT_SLURM_LOG_DIR:-${KD_GAT_LAKE_ROOT:-experimentruns}/slurm}"
mkdir -p "$SLURM_LOG_DIR"

JOB="${1:?Usage: scripts/submit.sh <job> [args...]}"
shift

# Read resource profile from YAML (single source of truth)
PROFILE=$(source .venv/bin/activate && python -m graphids submit-profile "$JOB")
read -r PARTITION CPUS MEM TIME SIGNAL MODE GRES COMMAND <<< "$PROFILE"

ACCT="--account=${KD_GAT_SLURM_ACCOUNT}"
PREAMBLE="source ${PROJECT_ROOT}/scripts/slurm/_preamble.sh"

ENV=""
case "$MODE" in
    cpu)     ENV="SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 " ;;
    cpu-raw) ENV="SKIP_CUDA_CONF=1 STAGE_DATA_ARGS=--raw " ;;
esac

GPU_ARGS=()
[[ -n "$GRES" && "$GRES" != "NONE" ]] && GPU_ARGS=(--gres="$GRES")

SIG_ARGS=()
[[ -n "$SIGNAL" && "$SIGNAL" != "NONE" ]] && SIG_ARGS=(--signal="B:${SIGNAL}")

sbatch "$ACCT" --partition="$PARTITION" --cpus-per-task="$CPUS" --mem="$MEM" \
    --time="$TIME" --job-name="graphids-${JOB}" "${GPU_ARGS[@]}" "${SIG_ARGS[@]}" \
    --output="${SLURM_LOG_DIR}/${JOB}_%j.out" --error="${SLURM_LOG_DIR}/${JOB}_%j.err" \
    --wrap="${ENV}${PREAMBLE} && ${COMMAND}$([ $# -gt 0 ] && printf ' %q' "$@")"
