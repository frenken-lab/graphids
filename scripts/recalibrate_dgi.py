"""Load existing DGI ckpt, fit SVDD centroid on train-normal, rescore val.

Bypasses the full orchestrator (which needs OTel + budget probe) and builds
the model + datamodule directly from the checkpoint and dataset config.
"""

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader as PyGDataLoader

from graphids.core.data.datamodule.graph import GraphDataModule
from graphids.core.data.datasets.can_bus import CANBusSource
from graphids.core.models.autoencoder.dgi_module import DGIModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- Load ckpt and reconstruct model from saved hyper_parameters ---
ckpt_path = "/fs/ess/PAS1266/graphids/dev/rf15/set_01/ablations/unsupervised/dgi/seed_42/checkpoints/best_model.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
hp = ckpt["hyper_parameters"]
print(f"ckpt epoch={ckpt.get('epoch')}  global_step={ckpt.get('global_step')}")
print(f"hyper_parameters keys: {sorted(hp.keys())}")

# Reconstruct module with the exact hp from training — num_ids > 0 triggers _build()
model = DGIModule(**hp)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print(f"loaded state_dict  missing={missing}  unexpected={unexpected}")
model.to(device)
model.eval()

# --- Build datamodule (lightweight — just loads cached tensors) ---
from graphids.config.constants import LAKE_ROOT

source = CANBusSource(name="set_01", lake_root=LAKE_ROOT)
dm = GraphDataModule(dataset=source, num_workers=0, dynamic_batching=False, label_filter="benign")
dm.setup("fit")
print(f"train graphs: {len(dm.train_dataset)}  val graphs: {len(dm.val_dataset)}")

# Simple fixed-batch-size loader (no budget probe needed for eval)
train_loader = PyGDataLoader(
    dm.train_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
val_loader = PyGDataLoader(
    dm.val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

# --- Calibrate SVDD centroid ---
print("calibrating SVDD centroid on train loader...")
model.calibrate_svdd_center(train_loader, device)
print(f"  centroid norm={model.svdd_center.norm().item():.4f}  dim={model.svdd_center.numel()}")

# --- Score val graphs ---
print("scoring val...")
scores_all = []
labels_all = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.clone().to(device)
        s = model._per_graph_scores(batch)
        scores_all.append(s.detach().cpu())
        labels_all.append(batch.y.detach().cpu())

scores = torch.cat(scores_all).numpy()
labels = torch.cat(labels_all).numpy()
print(f"n={len(labels)}  pos_rate={labels.mean():.4f}")
print(f"  score range [{scores.min():.4f}, {scores.max():.4f}]  mean={scores.mean():.4f}")

neg = scores[labels == 0]
pos = scores[labels == 1]
print(f"  benign  (n={len(neg)}): mean={neg.mean():.4f}  std={neg.std():.4f}")
print(f"  attacks (n={len(pos)}): mean={pos.mean():.4f}  std={pos.std():.4f}")
sep = (pos.mean() - neg.mean()) / (0.5 * (pos.std() + neg.std()))
print(f"  separation: {sep:+.3f} sigma")

auc = roc_auc_score(labels, scores)
ap = average_precision_score(labels, scores)
print(f"\nOCGIN-scored val-AUROC: {auc:.4f}  (v8 disc-score: 0.5201, v9-800ep disc: 0.4811)")
print(f"OCGIN-scored val-AP:    {ap:.4f}  (v8 disc-score: 0.1335, v9-800ep disc: 0.1087)")

# Save calibrated ckpt
out_path = ckpt_path.replace("best_model.ckpt", "best_model_ocgin.ckpt")
ckpt["state_dict"] = model.state_dict()
torch.save(ckpt, out_path)
print(f"\nsaved calibrated ckpt: {out_path}")
