#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=kd-gat-preprocess-test
#SBATCH --output=slurm_logs/preprocess_test_%j.out
#SBATCH --error=slurm_logs/preprocess_test_%j.err

# Preprocessing validation: builds CANBusDataset from raw data and checks feature dims.
# Stages raw data from NFS → scratch → TMPDIR, then runs preprocessing pipeline.
# Usage: sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/test_preprocessing.sbatch

SKIP_CUDA_CONF=1 STAGE_DATA_ARGS="--raw" source /users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh

log_job_header "Preprocessing test: hcrl_ch"

python -c "
import os
from graphids.core.preprocessing.datasets.can_bus import CANBusDataset
from graphids.config import cache_dir, data_dir

lake_root = os.environ.get('KD_GAT_LAKE_ROOT', 'experimentruns')
dataset = 'hcrl_ch'
root = cache_dir(lake_root, dataset)
raw = data_dir(lake_root, dataset)

print(f'lake_root: {lake_root}')
print(f'cache_dir: {root}')
print(f'data_dir:  {raw}')

ds = CANBusDataset(root=root, raw_dir=raw, split='train', window_size=100, stride=100, seed=42)
print(f'Dataset: {len(ds)} graphs')
assert len(ds) > 0, 'No graphs produced'

g = ds[0]
print(f'  x: {g.x.shape}, edge_index: {g.edge_index.shape}, edge_attr: {g.edge_attr.shape}, y: {g.y}')
assert g.x.shape[1] == 31, f'Expected 31 node features, got {g.x.shape[1]}'
assert g.edge_attr.shape[1] == 12, f'Expected 12 edge features, got {g.edge_attr.shape[1]}'

import torch
assert not torch.isnan(g.x).any(), 'NaN in node features'
assert not torch.isnan(g.edge_attr).any(), 'NaN in edge features'

print('PASS: Preprocessing produces valid graphs')
"

EXIT_CODE=$?
echo "Preprocessing test exit code: $EXIT_CODE"
exit $EXIT_CODE
