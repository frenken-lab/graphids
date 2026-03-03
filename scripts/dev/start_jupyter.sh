#!/usr/bin/env bash
# Start Jupyter Lab on a SLURM compute node with SSH tunnel instructions.
set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
VENV="/users/PAS2022/rf15/KD-GAT/.venv"
# Source .env for KD_GAT_SLURM_ACCOUNT
set -a; source "$PROJECT_ROOT/.env" 2>/dev/null; set +a
SLURM_ACCOUNT="${KD_GAT_SLURM_ACCOUNT:?Set KD_GAT_SLURM_ACCOUNT in .env}"
JUPYTER="$VENV/bin/jupyter"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
${BOLD}Usage:${NC} $(basename "$0") [OPTIONS]

Start Jupyter Lab on a SLURM compute node and print SSH tunnel instructions.

${BOLD}Options:${NC}
  -g, --gpu               Request GPU allocation (default)
  -c, --cpu-only          Request CPU-only allocation
  -t, --time <minutes>    Allocation time in minutes (default: 120)
  -p, --port <port>       Jupyter port (default: 8888)
  -h, --help              Show this help message

${BOLD}Examples:${NC}
  $(basename "$0")                    # GPU node, 2 hours, port 8888
  $(basename "$0") -c -t 60           # CPU-only, 1 hour
  $(basename "$0") -p 9999 -t 180     # GPU, 3 hours, port 9999

${BOLD}Alternative:${NC}
  OSC OnDemand (https://ondemand.osc.edu) provides a web-based Jupyter
  interface without SSH tunnels.
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
USE_GPU=true
TIME_MIN=120
PORT=8888

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpu)       USE_GPU=true;  shift ;;
        -c|--cpu-only)  USE_GPU=false; shift ;;
        -t|--time)      TIME_MIN="$2"; shift 2 ;;
        -p|--port)      PORT="$2";     shift 2 ;;
        -h|--help)      usage ;;
        *)              error "Unknown option: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [[ ! -x "$JUPYTER" ]]; then
    error "jupyter not found at $JUPYTER"
    echo ""
    echo "Install with:"
    echo "  source .venv/bin/activate"
    echo "  uv pip install jupyterlab"
    echo ""
    echo "Or use OSC OnDemand instead: https://ondemand.osc.edu"
    exit 1
fi

# ---------------------------------------------------------------------------
# Build SLURM job
# ---------------------------------------------------------------------------
SBATCH_ARGS=(
    --account="$SLURM_ACCOUNT"
    --time="$TIME_MIN"
    --cpus-per-task=4
    --mem=32G
    --job-name=jupyter
    --output="$PROJECT_ROOT/slurm_logs/jupyter-%j.out"
    --error="$PROJECT_ROOT/slurm_logs/jupyter-%j.err"
)

if $USE_GPU; then
    SBATCH_ARGS+=(--partition=gpu --gpus-per-node=1)
    info "Requesting GPU allocation"
else
    SBATCH_ARGS+=(--partition=cpu)
    info "Requesting CPU-only allocation"
fi

info "Time: ${TIME_MIN} minutes, Port: ${PORT}"

mkdir -p "$PROJECT_ROOT/slurm_logs"

# The Jupyter command to run on the compute node
JUPYTER_CMD="$JUPYTER lab --no-browser --ip=0.0.0.0 --port=$PORT --notebook-dir=$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Submit job
# ---------------------------------------------------------------------------
info "Submitting SLURM job..."
JOBID=$(sbatch "${SBATCH_ARGS[@]}" --parsable --wrap "$JUPYTER_CMD")

if [[ -z "$JOBID" ]]; then
    error "Failed to submit SLURM job."
    exit 1
fi

ok "Job submitted: $JOBID"
echo ""

# ---------------------------------------------------------------------------
# Wait for job to start
# ---------------------------------------------------------------------------
info "Waiting for job $JOBID to start..."
NODE=""
while true; do
    STATE=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null || echo "UNKNOWN")
    if [[ "$STATE" == "RUNNING" ]]; then
        NODE=$(squeue -j "$JOBID" -h -o "%N" 2>/dev/null)
        break
    elif [[ "$STATE" == "PENDING" ]]; then
        echo -ne "\r  Status: PENDING (waiting for resources)..."
        sleep 5
    elif [[ "$STATE" == "UNKNOWN" || -z "$STATE" ]]; then
        echo ""
        error "Job $JOBID is no longer in the queue. It may have failed."
        echo "Check: slurm_logs/jupyter-${JOBID}.err"
        exit 1
    else
        echo -ne "\r  Status: $STATE..."
        sleep 5
    fi
done

echo ""
ok "Job running on node: $NODE"
echo ""

# ---------------------------------------------------------------------------
# Print connection instructions
# ---------------------------------------------------------------------------
LOG_FILE="$PROJECT_ROOT/slurm_logs/jupyter-${JOBID}.out"

echo -e "${BOLD}=== SSH Tunnel Command ===${NC}"
echo ""
echo "  ssh -L $PORT:$NODE:$PORT rf15@pitzer.osc.edu"
echo ""
echo -e "${BOLD}=== Then open in browser ===${NC}"
echo ""
echo "  http://localhost:$PORT"
echo ""

# Wait a moment for Jupyter to start and write its log
info "Waiting for Jupyter to start..."
sleep 10

# Show the token URL from the log
if [[ -f "$LOG_FILE" ]]; then
    echo -e "${BOLD}=== Jupyter Log (token URL) ===${NC}"
    echo ""
    grep -m1 "http.*token=" "$LOG_FILE" 2>/dev/null || echo "  (token not yet available, check log: $LOG_FILE)"
    echo ""
fi

echo -e "${BOLD}=== Cleanup ===${NC}"
echo ""
echo "  scancel $JOBID"
echo ""
info "Log files:"
echo "  stdout: slurm_logs/jupyter-${JOBID}.out"
echo "  stderr: slurm_logs/jupyter-${JOBID}.err"
