#!/usr/bin/env bash
#SBATCH --clusters=pitzer
#SBATCH --account=pas1266
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=temporal-refactor-gate
set -euo pipefail
: "${REFACTOR_GATE_ROOT:?set REFACTOR_GATE_ROOT to a snapshot directory}"
: "${GRAPH_CODE_ROOT:?set GRAPH_CODE_ROOT to the source checkout}"
export GRAPHIDS_LAKE_ROOT="${GRAPHIDS_LAKE_ROOT:-/fs/ess/PAS1266/graphids}"
export GRAPHIDS_RUN_ROOT="${GRAPHIDS_RUN_ROOT:-/fs/ess/PAS1266/graphids/dev/rf15}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python="$GRAPH_CODE_ROOT/.venv/bin/python"
cd "$REFACTOR_GATE_ROOT/current"
"$python" -m pytest "$GRAPH_CODE_ROOT/tests/core/models/test_temporal_cuda_graph.py" "$GRAPH_CODE_ROOT/tests/core/models/test_temporal_hybrid.py" -q > "$REFACTOR_GATE_ROOT/cuda-correctness.txt" 2>&1
for pair in 1 2 3; do
    if (( pair % 2 )); then order=(before current); else order=(current before); fi
    for version in "${order[@]}"; do
        cd "$REFACTOR_GATE_ROOT/$version"
        "$python" -m graphids.exp.temporal_comparison "$REFACTOR_GATE_ROOT/workload.yml" \
            --output "$REFACTOR_GATE_ROOT/pair${pair}_${version}" --section-limit 2048 \
            > "$REFACTOR_GATE_ROOT/pair${pair}_${version}.log" 2>&1
    done
done
cd "$REFACTOR_GATE_ROOT/current"
"$python" -m graphids.exp.temporal_cuda_graph "$REFACTOR_GATE_ROOT/eager-graph" \
    > "$REFACTOR_GATE_ROOT/eager-graph.log" 2>&1
"$python" - "$REFACTOR_GATE_ROOT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

import torch

from graphids.exp.trials import validate_trial

root = Path(sys.argv[1])
pairs = []

def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b

for index in range(1, 4):
    before = json.loads((root / f"pair{index}_before/comparison.json").read_text())
    current = json.loads((root / f"pair{index}_current/comparison.json").read_text())
    validate_trial(current, before)
    if before["optimizer_steps"] != current["optimizer_steps"]:
        raise ValueError("optimizer update counts differ")
    if current["cuda_graph"].get("disabled") or not current["cuda_graph"].get("replays"):
        raise ValueError("refactored capture did not replay")
    old_checkpoint = torch.load(root / f"pair{index}_before/checkpoints/final.ckpt", map_location="cpu", weights_only=False)
    new_checkpoint = torch.load(root / f"pair{index}_current/checkpoints/final.ckpt", map_location="cpu", weights_only=False)
    if not all(equal(old_checkpoint[key], new_checkpoint[key]) for key in ("state_dict", "optimizer_states")):
        raise ValueError("model or optimizer updates differ")
    pairs.append(current["end_to_end_scored_events_per_second"] / before["end_to_end_scored_events_per_second"] - 1)
median = statistics.median(pairs)
separate = json.loads((root / "eager-graph/pairs.json").read_text())["selection"]
report = {"paired_changes": pairs, "median_change": median,
          "refactor_accepted": median >= -0.03,
          "eager_graph_selection": separate}
(root / "gate.json").write_text(json.dumps(report, indent=2) + "\n")
if not report["refactor_accepted"] or not separate["accepted"]:
    raise SystemExit("CUDA graph gate failed: " + json.dumps(report))
print(json.dumps(report))
PY
