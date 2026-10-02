#!/usr/bin/env bash
#SBATCH --clusters=pitzer
#SBATCH --account=pas1266
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=temporal-graph-epoch
set -euo pipefail
cd "${GRAPH_CODE_ROOT:-/users/PAS2022/rf15/projects/lab/graphids}"
source scripts/slurm/_preamble.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${GRAPH_ROOT:?set GRAPH_ROOT to a unique shared output directory}"
: "${GRAPH_EPOCH_CONFIG:?set GRAPH_EPOCH_CONFIG to the reviewed full-coverage configuration}"
if [[ -n "${GRAPH_GATE_REPORT:-}" ]]; then
    python - "$GRAPH_GATE_REPORT" <<'PYGATE'
import json
import sys
from pathlib import Path
assert json.loads(Path(sys.argv[1]).read_text())["selection"]["accepted"], "bounded CUDA graph gate failed"
PYGATE
fi
python -c 'import torch; assert torch.cuda.is_available(); assert "V100" in torch.cuda.get_device_name()'
python -m graphids.exp.temporal_comparison "$GRAPH_EPOCH_CONFIG" --output "$GRAPH_ROOT"
