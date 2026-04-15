#!/usr/bin/env bash
# Unified SLURM job launcher. Run from login node.
# Submit profiles are read from configs/resources/submit_profiles.json via Python.
#
# Usage:
#   scripts/slurm/submit.sh tests [-k pattern] [-x]
#   scripts/slurm/submit.sh rebuild-caches [--dataset hcrl_ch | --all] [--delete-existing]
#   scripts/slurm/submit.sh validate
#   scripts/slurm/submit.sh analyze --ckpt-path <path> --dataset <name> [--cka-teacher-ckpt <p>]
#   scripts/slurm/submit.sh ablation [--recipe X --dataset X --seed X]
#   scripts/slurm/submit.sh profile [stage scale dataset]
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

# Sniff --dataset and --scale from args so submit-profile can auto-size.
# Non-destructive: args are forwarded to the downstream command unchanged.
DATASET=""
SCALE=""
for (( i=1; i<=$#; i++ )); do
    case "${!i}" in
        --dataset) j=$((i+1)); DATASET="${!j:-}" ;;
        --dataset=*) DATASET="${!i#--dataset=}" ;;
        --scale) j=$((i+1)); SCALE="${!j:-}" ;;
        --scale=*) SCALE="${!i#--scale=}" ;;
    esac
done

PROFILE_ARGS=("$JOB")
[[ -n "$DATASET" ]] && PROFILE_ARGS+=(--dataset "$DATASET")
[[ -n "$SCALE" ]] && PROFILE_ARGS+=(--scale "$SCALE")

# Read resource profile (auto-sized from dataset + scale when provided)
PROFILE=$(source .venv/bin/activate && python -m graphids submit-profile "${PROFILE_ARGS[@]}")
read -r PARTITION CPUS MEM TIME SIGNAL MODE GRES COMMAND <<< "$PROFILE"

ACCT="--account=${GRAPHIDS_SLURM_ACCOUNT}"
PREAMBLE="source ${SCRIPT_DIR}/_preamble.sh"

ENV=""
[[ "$MODE" == "cpu" ]] && ENV="SKIP_CUDA_CONF=1 "

GPU_ARGS=()
[[ -n "$GRES" && "$GRES" != "NONE" ]] && GPU_ARGS=(--gres="$GRES")

SIG_ARGS=()
[[ -n "$SIGNAL" && "$SIGNAL" != "NONE" ]] && SIG_ARGS=(--signal="B:${SIGNAL}")

# Optional: SBATCH_DEP="afterok:<jobid>[:<jobid>...]" chains jobs.
DEP_ARGS=()
[[ -n "${SBATCH_DEP:-}" ]] && DEP_ARGS=(--dependency="$SBATCH_DEP")

sbatch "$ACCT" --partition="$PARTITION" --cpus-per-task="$CPUS" --mem="$MEM" \
    --time="$TIME" --job-name="graphids-${JOB}" "${GPU_ARGS[@]}" "${SIG_ARGS[@]}" "${DEP_ARGS[@]}" \
    --output="${SLURM_LOG_DIR}/${JOB}_%j.out" --error="${SLURM_LOG_DIR}/${JOB}_%j.err" \
    --wrap="${ENV}${PREAMBLE} && ${COMMAND}$([ $# -gt 0 ] && printf ' %q' "$@")"
