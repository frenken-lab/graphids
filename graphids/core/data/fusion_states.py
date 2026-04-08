"""Extract and cache fusion state vectors from upstream VGAE + GAT checkpoints.

Short GPU job (~2 min). Saves cached states to disk so fusion training
can run on CPU without loading upstream models.

CLI surface: ``python -m graphids extract-fusion-states``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter

from graphids.core.data.sampler import make_graph_loader
from graphids.log import get_logger

log = get_logger(__name__)

FUSION_STATES_DIR = "fusion_states"
TRAIN_FILENAME = "train_states.pt"
VAL_FILENAME = "val_states.pt"


# ---------------------------------------------------------------------------
# Feature extractors — produce fixed-size feature vectors from model output.
# Order is load-bearing: matches the 15-D layout baked into trained
# DQN/bandit checkpoints (VGAE 8-D then GAT 7-D).
# ---------------------------------------------------------------------------


class VGAEFusionExtractor:
    """Extract 8-D features from VGAE output.

    Layout: [0:3] errors (node recon, neighbor, canid)
            [3:7] latent stats (mean, std, max, min)
            [7]   confidence (1 / (1 + recon_err))
    """

    feature_dim = 8
    confidence_index = 7

    def extract(self, model: torch.nn.Module, batch, device: torch.device) -> torch.Tensor:
        edge_attr = (
            getattr(batch, "edge_attr", None) if getattr(model, "_uses_edge_attr", False) else None
        )
        cont, canid_logits, nbr_logits, z, _, _ = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            node_id=batch.node_id,
        )
        b = batch.batch

        node_sq_err = (cont - batch.x).pow(2).mean(dim=1)
        recon_err = scatter(node_sq_err, b, dim=0, reduce="mean")

        canid_ce = F.cross_entropy(canid_logits, batch.node_id, reduction="none")
        canid_err = scatter(canid_ce, b, dim=0, reduce="mean")

        nbr_targets = model.create_neighborhood_targets(batch.node_id, batch.edge_index, b)
        nbr_bce = F.binary_cross_entropy_with_logits(
            nbr_logits, nbr_targets, reduction="none"
        ).mean(dim=1)
        nbr_err = scatter(nbr_bce, b, dim=0, reduce="mean")

        z_mean = scatter(z.mean(dim=1), b, dim=0, reduce="mean")
        z_std = scatter(z.std(dim=1), b, dim=0, reduce="mean")
        z_max = scatter(z.max(dim=1).values, b, dim=0, reduce="max")
        z_min = scatter(z.min(dim=1).values, b, dim=0, reduce="min")

        conf = 1.0 / (1.0 + recon_err)

        return torch.stack(
            [recon_err, nbr_err, canid_err, z_mean, z_std, z_max, z_min, conf], dim=1
        )


class GATFusionExtractor:
    """Extract 7-D features from GAT output.

    Layout: [0:2] class probabilities (class 0, class 1)
            [2:6] embedding stats (mean, std, max, min)
            [6]   confidence (1 - normalized entropy)
    """

    feature_dim = 7
    confidence_index = 6

    def extract(self, model: torch.nn.Module, batch, device: torch.device) -> torch.Tensor:
        logits, emb = model(batch, return_embedding=True)

        probs = F.softmax(logits, dim=1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
        conf = (1.0 - entropy / math.log(2)).clamp(0.0, 1.0)

        emb_mean = emb.mean(dim=1)
        emb_std = emb.std(dim=1)
        emb_max = emb.max(dim=1).values
        emb_min = emb.min(dim=1).values

        return torch.cat(
            [
                probs,
                emb_mean.unsqueeze(1),
                emb_std.unsqueeze(1),
                emb_max.unsqueeze(1),
                emb_min.unsqueeze(1),
                conf.unsqueeze(1),
            ],
            dim=1,
        )


EXTRACTORS: dict[str, VGAEFusionExtractor | GATFusionExtractor] = {
    "vgae": VGAEFusionExtractor(),
    "gat": GATFusionExtractor(),
}


def cache_predictions(
    models: dict[str, torch.nn.Module],
    data,
    device: torch.device,
    max_samples: int = 150_000,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Run registered extractors over data, produce N-D state vectors for fusion."""
    active = [(name, ext) for name, ext in EXTRACTORS.items() if name in models]
    for model in models.values():
        model.eval()

    capped = data[:max_samples]
    loader = make_graph_loader(capped, batch_size=batch_size)

    states, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            feats = [ext.extract(models[name], batch, device) for name, ext in active]
            states.append(torch.cat(feats, dim=1))
            labels.append(batch.y)

    return {"states": torch.cat(states), "labels": torch.cat(labels)}


def extract_fusion_states(
    *,
    vgae_ckpt: str,
    gat_ckpt: str,
    dataset: str,
    output_dir: str,
    max_samples: int = 150_000,
    max_val_samples: int = 30_000,
    batch_size: int = 256,
    seed: int = 42,
    window_size: int = 100,
    stride: int = 100,
    val_fraction: float = 0.2,
) -> None:
    """Load VGAE + GAT checkpoints, cache fusion states to ``output_dir``."""
    from graphids.core.data.datamodule.graph import load_datasets
    from graphids.core.data.datasets.can_bus import CANBusDataset
    from graphids.core.models.base import load_inner_model

    lake_root = os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("loading_upstream_models", vgae=vgae_ckpt, gat=gat_ckpt, device=str(device))
    vgae, _ = load_inner_model("vgae", Path(vgae_ckpt), device)
    gat, _ = load_inner_model("gat", Path(gat_ckpt), device)
    models = {"vgae": vgae, "gat": gat}

    train_ds, val_ds, _ = load_datasets(
        dataset=dataset,
        lake_root=lake_root,
        seed=seed,
        window_size=window_size,
        stride=stride,
        train_val_split=1.0 - val_fraction,
        dataset_cls=CANBusDataset,
    )

    log.info("extracting_train_states", n_graphs=len(train_ds), max_samples=max_samples)
    train_cache = cache_predictions(
        models,
        list(train_ds),
        device,
        max_samples,
        batch_size=batch_size,
    )

    log.info("extracting_val_states", n_graphs=len(val_ds), max_samples=max_val_samples)
    val_cache = cache_predictions(
        models,
        list(val_ds),
        device,
        max_val_samples,
        batch_size=batch_size,
    )

    # Move to CPU for disk serialization
    train_cache = {k: v.cpu() for k, v in train_cache.items()}
    val_cache = {k: v.cpu() for k, v in val_cache.items()}

    out = Path(output_dir) / FUSION_STATES_DIR
    out.mkdir(parents=True, exist_ok=True)
    torch.save(train_cache, out / TRAIN_FILENAME)
    torch.save(val_cache, out / VAL_FILENAME)

    log.info(
        "fusion_states_saved",
        output_dir=str(out),
        train_states=list(train_cache["states"].shape),
        val_states=list(val_cache["states"].shape),
    )
