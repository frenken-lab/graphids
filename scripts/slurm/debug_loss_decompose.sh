#!/usr/bin/env bash
# Decompose VGAE loss into components to identify which term is exploding
#SBATCH --partition=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --job-name=kd-gat-loss
#SBATCH --output=slurm_logs/loss_%j.out
#SBATCH --error=slurm_logs/loss_%j.err

set -euo pipefail
cd "/users/PAS2022/rf15/KD-GAT"
mkdir -p slurm_logs
module load python/3.12
source .venv/bin/activate
set -a; source .env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/data/stage_data.sh --cache

echo "=== Loss Decomposition ==="
python -c "
import torch
import torch.nn.functional as F
from graphids.config import resolve, data_dir, cache_dir
from graphids.pipeline.stages.data_loading import load_data, make_dataloader
from graphids.pipeline.stages.modules import VGAEModule

cfg = resolve('vgae', 'large', dataset='hcrl_ch')
train_data, val_data, num_ids, in_ch = load_data(cfg)
print(f'Data: {len(train_data)} train, {len(val_data)} val, {num_ids} ids, {in_ch} features')

module = VGAEModule(cfg, num_ids, in_ch)
module = module.cuda()
module.eval()

dl = make_dataloader(val_data, cfg, batch_size=32, shuffle=False)
batch = next(iter(dl)).cuda()
print(f'Batch: {batch.num_graphs} graphs, {batch.x.shape[0]} nodes, features shape {batch.x.shape}')
print(f'Feature 0 (CAN_ID): min={batch.x[:,0].min():.1f} max={batch.x[:,0].max():.1f}')

with torch.no_grad():
    cont_out, canid_logits, nbr_logits, z, kl_loss = module(batch)

    print(f'\nModel outputs:')
    print(f'  cont_out: shape={cont_out.shape}, min={cont_out.min():.6f}, max={cont_out.max():.6f}')
    print(f'  canid_logits: shape={canid_logits.shape}, min={canid_logits.min():.4f}, max={canid_logits.max():.4f}')
    print(f'  nbr_logits: shape={nbr_logits.shape}, min={nbr_logits.min():.4f}, max={nbr_logits.max():.4f}')
    print(f'  z: shape={z.shape}, min={z.min():.4f}, max={z.max():.4f}, mean={z.mean():.4f}')
    print(f'  kl_loss: {kl_loss.item():.4f}')

    recon = F.mse_loss(cont_out, batch.x[:, 1:])
    canid = F.cross_entropy(canid_logits, batch.x[:, 0].long())
    nbr_targets = module.model.create_neighborhood_targets(batch.x, batch.edge_index, batch.batch)
    nbr_loss = F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets)

    print(f'\nLoss components:')
    print(f'  recon (MSE):      {recon.item():.6f}')
    print(f'  canid (CE):       {canid.item():.4f} (weighted 0.1x = {0.1*canid.item():.4f})')
    print(f'  nbr (BCE):        {nbr_loss.item():.4f} (weighted 0.05x = {0.05*nbr_loss.item():.4f})')
    print(f'  kl:               {kl_loss.item():.4f} (weighted 0.01x = {0.01*kl_loss.item():.4f})')

    total = recon + 0.1 * canid + 0.05 * nbr_loss + 0.01 * kl_loss
    print(f'  TOTAL:            {total.item():.4f}')

    # Check fp16 overflow potential
    max_abs = max(abs(recon.item()), abs(0.1*canid.item()), abs(0.05*nbr_loss.item()), abs(0.01*kl_loss.item()))
    print(f'\n  Max component magnitude: {max_abs:.4f}')
    print(f'  fp16 max: 65504')
    print(f'  Would overflow fp16: {max_abs > 65504}')
"
