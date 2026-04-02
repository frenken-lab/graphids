#!/usr/bin/env bash
# Unified SLURM job launcher. Run from login node.
# Resource profiles are read from graphids/config/resources.yaml via Python.
#
# Usage:
#   scripts/submit.sh tests [-k pattern] [-x]
#   scripts/submit.sh rebuild-caches [--dataset hcrl_ch | --all] [--delete-existing]
#   scripts/submit.sh validate
#   scripts/submit.sh landscape <model_type> <dataset> <ckpt_path> [--resolution N]
#   scripts/submit.sh preprocessing-test [--dataset hcrl_ch]
#   scripts/submit.sh ablation [--recipe X --dataset X --seed X]
#   scripts/submit.sh profile [stage scale dataset]
set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
cd "$PROJECT_ROOT"
source .env
mkdir -p slurm_logs

JOB="${1:?Usage: scripts/submit.sh <job> [args...]}"
shift

# Read resource profile from YAML (single source of truth)
PROFILE=$(source .venv/bin/activate && python -m graphids submit-profile "$JOB")
read -r PARTITION CPUS MEM TIME SIGNAL MODE COMMAND <<< "$PROFILE"

ACCT="--account=${KD_GAT_SLURM_ACCOUNT}"
PREAMBLE="source ${PROJECT_ROOT}/scripts/slurm/_preamble.sh"

ENV=""
case "$MODE" in
    cpu)     ENV="SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 " ;;
    cpu-raw) ENV="SKIP_CUDA_CONF=1 STAGE_DATA_ARGS=--raw " ;;
esac

SIG_ARGS=()
[[ -n "$SIGNAL" && "$SIGNAL" != "NONE" ]] && SIG_ARGS=(--signal="B:${SIGNAL}")

sbatch "$ACCT" --partition="$PARTITION" --cpus-per-task="$CPUS" --mem="$MEM" \
    --time="$TIME" --job-name="kd-gat-${JOB}" "${SIG_ARGS[@]}" \
    --output="slurm_logs/${JOB}_%j.out" --error="slurm_logs/${JOB}_%j.err" \
    --wrap="${ENV}${PREAMBLE} && ${COMMAND} $(printf '%q ' "$@")"
