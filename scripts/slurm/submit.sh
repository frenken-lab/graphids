#!/usr/bin/env bash
# Unified SLURM job launcher. Run from login node.
# Submit profiles are read from configs/resources/submit_profiles.json via Python.
#
# Usage:
#   scripts/slurm/submit.sh tests [-k pattern] [-x]
#   scripts/slurm/submit.sh rebuild-caches [--dataset hcrl_ch | --all] [--delete-existing]
#   scripts/slurm/submit.sh validate
#   scripts/slurm/submit.sh landscape <model_type> <dataset> <ckpt_path> [--resolution N]
#   scripts/slurm/submit.sh ablation [--recipe X --dataset X --seed X]
#   scripts/slurm/submit.sh profile [stage scale dataset]
#   scripts/slurm/submit.sh probe-budget [args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
source .env
SLURM_LOG_DIR="${GRAPHIDS_SLURM_LOG_DIR:-${GRAPHIDS_LAKE_ROOT:-experimentruns}/slurm}"
mkdir -p "$SLURM_LOG_DIR"

command -v jsonnet >/dev/null 2>&1 || {
    echo "submit.sh: jsonnet binary not found on PATH." >&2
    echo "  Install go-jsonnet 0.20.0+ to ~/.local/bin/ — see ADR 0010 in docs/decisions/README.md" >&2
    exit 1
}

JOB="${1:?Usage: scripts/slurm/submit.sh <job> [args...]}"
shift

# Read resource profile from YAML (single source of truth)
PROFILE=$(source .venv/bin/activate && python -m graphids submit-profile "$JOB")
read -r PARTITION CPUS MEM TIME SIGNAL MODE GRES COMMAND <<< "$PROFILE"

ACCT="--account=${GRAPHIDS_SLURM_ACCOUNT}"
PREAMBLE="source ${SCRIPT_DIR}/_preamble.sh"

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
