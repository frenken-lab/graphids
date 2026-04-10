"""2D loss surfaces around trained model minima (Li et al., 2018).

Generates a grid of loss values by perturbing weights along two
filter-normalized random directions. Output is a Parquet file
suitable for contour/heatmap visualization in the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader as PyGDataLoader
from graphids._otel import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Filter-normalized random directions (Li et al., 2018)
# ---------------------------------------------------------------------------


def _filter_normalize(
    direction: list[torch.Tensor],
    reference: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Normalize each direction tensor to match the reference parameter's norm.

    For 2D+ tensors, normalizes per-filter (row). For 1D, normalizes globally.
    """
    normalized = []
    for d, r in zip(direction, reference):
        if d.dim() >= 2:
            d_flat = d.reshape(d.shape[0], -1)
            r_flat = r.reshape(r.shape[0], -1)
            r_norms = r_flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
            d_norms = d_flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
            normalized.append((d_flat * (r_norms / d_norms)).reshape(d.shape))
        else:
            r_norm = r.norm().clamp(min=1e-10)
            d_norm = d.norm().clamp(min=1e-10)
            normalized.append(d * (r_norm / d_norm))
    return normalized


def _random_direction(model: torch.nn.Module, seed: int) -> list[torch.Tensor]:
    """Generate a single filter-normalized random direction."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    params = [p.data for p in model.parameters()]
    raw = [torch.randn(p.shape, generator=rng, dtype=p.dtype).to(p.device) for p in params]
    return _filter_normalize(raw, params)


def _perturb_model(
    model: torch.nn.Module,
    base_params: list[torch.Tensor],
    dir1: list[torch.Tensor],
    dir2: list[torch.Tensor],
    alpha: float,
    beta: float,
) -> None:
    """Set model parameters to base + alpha*dir1 + beta*dir2 (in-place)."""
    for p, b, d1, d2 in zip(model.parameters(), base_params, dir1, dir2):
        p.data.copy_(b + alpha * d1 + beta * d2)


# ---------------------------------------------------------------------------
# Model-specific loss functions
# ---------------------------------------------------------------------------


@torch.no_grad()
def _vgae_loss(model, dataloader, device: torch.device, cfg) -> float:
    """VGAE reconstruction + CAN ID + neighborhood + KL loss (from config weights)."""
    model.eval()
    total, count = 0.0, 0
    for batch in dataloader:
        batch = batch.clone().to(device)
        edge_attr = getattr(batch, "edge_attr", None)
        cont, canid_logits, nbr_logits, _z, kl_loss, _ = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            node_id=batch.node_id,
        )
        recon = F.mse_loss(cont, batch.x)
        canid = F.cross_entropy(canid_logits, batch.node_id)
        nbr_targets = model.create_neighborhood_targets(
            batch.node_id, batch.edge_index, batch.batch
        )
        nbr = F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets)
        loss = recon + cfg.canid_weight * canid + cfg.nbr_weight * nbr + cfg.kl_weight * kl_loss
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return total / max(count, 1)


@torch.no_grad()
def _gat_loss(model, dataloader, device: torch.device, _cfg) -> float:
    """GAT cross-entropy loss."""
    model.eval()
    total, count = 0.0, 0
    for batch in dataloader:
        batch = batch.clone().to(device)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y)
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return total / max(count, 1)


@torch.no_grad()
def _dgi_loss(model, dataloader, device: torch.device, _cfg) -> float:
    """DGI contrastive mutual-information loss (real vs shuffled node features)."""
    model.eval()
    total, count = 0.0, 0
    for batch in dataloader:
        batch = batch.clone().to(device)
        edge_attr = getattr(batch, "edge_attr", None)
        pos_z, neg_z, summary = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            node_id=batch.node_id,
        )
        loss = model.dgi_loss(pos_z, neg_z, summary, batch.batch)
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return total / max(count, 1)


_LOSS_FN = {"vgae": _vgae_loss, "gat": _gat_loss, "dgi": _dgi_loss}


# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------


def _sweep_grid(
    model: torch.nn.Module,
    loss_fn,
    dataloader,
    device: torch.device,
    cfg,
    resolution: int,
    scale: float,
    seed: int,
) -> dict:
    """Compute loss on a resolution x resolution grid of perturbations."""
    dir1 = _random_direction(model, seed)
    dir2 = _random_direction(model, seed + 1)
    base_params = [p.data.clone() for p in model.parameters()]

    alphas = np.linspace(-scale, scale, resolution)
    betas = np.linspace(-scale, scale, resolution)
    total = resolution * resolution

    xs, ys, losses = [], [], []
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            _perturb_model(model, base_params, dir1, dir2, a, b)
            losses.append(loss_fn(model, dataloader, device, cfg))
            xs.append(float(a))
            ys.append(float(b))

            done = i * resolution + j + 1
            if done % max(1, total // 10) == 0:
                log.info("loss_landscape_progress", done=done, total=total)

    # Restore original parameters
    _perturb_model(model, base_params, dir1, dir2, 0.0, 0.0)
    return {"x": xs, "y": ys, "loss": losses}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_and_save_loss_landscape(
    model: torch.nn.Module,
    model_type: str,
    val_data: list,
    device: torch.device,
    output_dir: Path,
    hparams,
    *,
    resolution: int = 51,
    scale: float = 1.0,
    seed: int = 42,
    max_graphs: int = 500,
    dataset: str = "",
) -> None:
    """Compute loss landscape for a model and save as Parquet."""
    loss_fn = _LOSS_FN.get(model_type)
    if loss_fn is None:
        log.warning("loss_landscape_skip", model_type=model_type, reason="no loss function")
        return

    if len(val_data) > max_graphs:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(val_data), max_graphs, replace=False)
        data = [val_data[i] for i in indices]
    else:
        data = val_data

    dataloader = PyGDataLoader(data, batch_size=min(256, len(data)))

    log.info("loss_landscape_start", model=model_type, resolution=resolution, scale=scale)
    result = _sweep_grid(model, loss_fn, dataloader, device, hparams, resolution, scale, seed)
    _save_parquet(result, model_type, dataset, output_dir)


def _save_parquet(result: dict, model_type: str, dataset: str, output_dir: Path) -> None:
    """Save loss landscape grid as Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    result["model_type"] = [model_type] * len(result["x"])
    result["dataset"] = [dataset] * len(result["x"])

    path = output_dir / f"loss_landscape_{model_type}.parquet"
    pq.write_table(pa.table(result), path)
    log.info("loss_landscape_saved", model=model_type, points=len(result["x"]), path=str(path))
