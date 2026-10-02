#!/usr/bin/env bash
#SBATCH --clusters=pitzer
#SBATCH --account=pas1266
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=temporal-timeline-256
set -euo pipefail
cd "${PROFILE_CODE_ROOT:-/users/PAS2022/rf15/projects/lab/graphids}"
source scripts/slurm/_diagnostic.sh
: "${PROFILE_ROOT:?set PROFILE_ROOT to a unique output directory}"
diagnostic_init "$PROFILE_ROOT"
python -m pytest tests/exp/test_temporal_profile.py tests/core/models/test_temporal_hybrid.py -q > "$PROFILE_ROOT/correctness.txt" 2>&1
config="$PROFILE_ROOT/two-workers.yml"
python - "$config" <<'PYCONFIG'
import sys
from pathlib import Path
import yaml
from graphids.exp.config import load_experiment_yaml
base = Path("configs/experiments/comparisons/temporal_set_01_segmented_lanes256_scaling_diagnostic.yml")
cfg = load_experiment_yaml(base)
cfg["config"]["data"].update(num_workers=2, prebuild_batches=2, persistent_workers=True)
Path(sys.argv[1]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PYCONFIG
python -m graphids.exp.temporal_comparison "$config" --output "$PROFILE_ROOT/baseline" --section-limit 256
python -m graphids.exp.temporal_comparison "$config" --output "$PROFILE_ROOT/instrumented" --section-limit 256 --profile
