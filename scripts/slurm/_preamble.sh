#!/usr/bin/env bash
# scripts/slurm/_preamble.sh — sourced by SLURM job scripts.
# Sets up Python environment + env vars.
#
# Override before sourcing:
#   SKIP_CUDA_CONF=1   — skip CUDA alloc config (CPU-only jobs)

set -euo pipefail

# Without this, Python fully buffers stdout under SLURM (no TTY).
# turm and tail -f see nothing until the process exits.
export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

module load python/3.12
source .venv/bin/activate
set -a; source ./.env; set +a

SLURM_LOG_DIR="${GRAPHIDS_SLURM_LOG_DIR:-${GRAPHIDS_LAKE_ROOT:-experimentruns}/slurm}"
mkdir -p "$SLURM_LOG_DIR"

# Group-writable umask for shared ESS data lake
umask 002

if [[ "${SKIP_CUDA_CONF:-0}" != "1" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
    # torch.compiler.reset() only clears dynamo state, not the inductor
    # disk cache. Stale artifacts from prior jobs on shared nodes can
    # cause compilation failures. (pytorch/pytorch#172024)
    rm -rf "/tmp/torchinductor_${USER:-$LOGNAME}"
fi

# NOTE: `python -m graphids stage-data` (NFS → scratch → TMPDIR) used to
# run here. The command was deleted in an earlier refactor; the call
# itself remained because its `grep '^export '` pipe swallowed the
# error. Rebuild + training jobs read directly from ESS NFS and work
# fine. If training ever gets I/O-bound, reintroduce a real stage-data
# command — don't paper over with another silent eval.

if [[ -n "${TMPDIR:-}" ]]; then
    export GRAPHIDS_STAGE_DIR="$TMPDIR/graphids-stage"
    mkdir -p "$GRAPHIDS_STAGE_DIR"
fi
