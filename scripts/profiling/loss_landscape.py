"""Compute 2D loss surfaces around trained model minima.

For each model checkpoint, generates a grid of loss values by perturbing
weights along two filter-normalized random directions (Li et al., 2018).
Output is a Parquet file suitable for contour/heatmap visualization.

Usage:
    python scripts/profiling/loss_landscape.py \
        --model vgae --dataset hcrl_sa [--resolution 51] [--scale 1.0]

Must run on GPU node (SLURM).
"""

from __future__ import annotations

import argparse
import structlog
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from graphids.config import NODE_FEATURE_COUNT, resolve

log = structlog.get_logger()

LAKE_ROOT = Path(os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns"))
OUTPUT_ROOT = Path("data/loss_landscapes")

# Map model_type → (stage subdirectory suffix, scale)
_MODEL_RUN_MAP = {
    "vgae": ("vgae_large_autoencoder", "large"),
    "gat": ("gat_large_curriculum", "large"),
    "dqn": ("dqn_large_fusion", "large"),
}


# ---------------------------------------------------------------------------
# Filter-normalized random directions (Li et al., 2018)
# ---------------------------------------------------------------------------


def _filter_normalize(
    direction: list[torch.Tensor], reference: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Normalize each direction tensor to have the same norm as the corresponding reference parameter.

    For conv filters (4-D), normalizes per-filter. For other tensors, normalizes globally.
    """
    normalized = []
    for d, r in zip(direction, reference):
        if d.dim() >= 2:
            # Per-filter normalization: reshape to (num_filters, -1)
            d_flat = d.reshape(d.shape[0], -1)
            r_flat = r.reshape(r.shape[0], -1)
            r_norms = r_flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
            d_norms = d_flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
            d_flat = d_flat * (r_norms / d_norms)
            normalized.append(d_flat.reshape(d.shape))
        else:
            r_norm = r.norm().clamp(min=1e-10)
            d_norm = d.norm().clamp(min=1e-10)
            normalized.append(d * (r_norm / d_norm))
    return normalized


def _random_direction(model: torch.nn.Module, seed: int) -> list[torch.Tensor]:
    """Generate a single filter-normalized random direction."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    params = [p.data for p in model.parameters()]
    # Generate on CPU (generator only works on CPU), then move to param device
    raw = [torch.randn(p.shape, generator=rng, dtype=p.dtype).to(p.device) for p in params]
    return _filter_normalize(raw, params)


# ---------------------------------------------------------------------------
# Loss evaluation
# ---------------------------------------------------------------------------


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


def _evaluate_vgae_loss(model, dataloader, device: torch.device) -> float:
    """Evaluate VGAE reconstruction + KL loss on data."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.clone().to(device)
            edge_attr = getattr(batch, "edge_attr", None)
            cont_out, canid_logits, nbr_logits, _z, kl_loss, _ = model(
                batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr,
                node_id=batch.node_id,
            )
            recon = F.mse_loss(cont_out, batch.x)
            canid = F.cross_entropy(canid_logits, batch.node_id)
            nbr_targets = model.create_neighborhood_targets(batch.node_id, batch.edge_index, batch.batch)
            nbr_loss = F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets)
            loss = recon + 0.1 * canid + 0.05 * nbr_loss + 0.01 * kl_loss
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs
    return total_loss / max(count, 1)


def _evaluate_gat_loss(model, dataloader, device: torch.device) -> float:
    """Evaluate GAT cross-entropy loss on data."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.clone().to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs
    return total_loss / max(count, 1)


def _evaluate_dqn_loss(model, dataloader, device: torch.device) -> float:
    """Evaluate DQN Q-surface smoothness for loss landscape visualization.

    Computes a self-consistency surrogate: Huber loss between Q-values and
    their own per-state max (the greedy target). This measures how the
    Q-surface varies around the trained minimum, not the actual training loss.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.clone().to(device)
            # DQN Q-network takes state features, not graph batches
            # Use a simple forward pass and measure deviation from minimum Q-values
            q_values = model(batch)
            # Self-consistency loss: how much Q-values deviate from their own targets
            target = q_values.detach().max(dim=1, keepdim=True)[0].expand_as(q_values)
            loss = F.smooth_l1_loss(q_values, target)
            total_loss += loss.item() * batch.shape[0]
            count += batch.shape[0]
    return total_loss / max(count, 1)


_LOSS_FN = {
    "vgae": _evaluate_vgae_loss,
    "gat": _evaluate_gat_loss,
    "dqn": _evaluate_dqn_loss,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_graph_data(dataset: str, model_type: str, cfg, max_graphs: int = 500):
    """Load preprocessed graph data for loss evaluation."""
    from graphids.core.preprocessing import CANBusDataModule

    dm = CANBusDataModule.from_cfg(cfg)
    dm.setup("fit")
    graphs = list(dm.train_dataset) + list(dm.val_dataset)
    # Subsample for efficiency
    if len(graphs) > max_graphs:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(graphs), max_graphs, replace=False)
        graphs = [graphs[i] for i in indices]
    return graphs


def _make_dataloader(graphs, cfg):
    """Create a PyG DataLoader from graph list."""
    from torch_geometric.loader import DataLoader as PyGDataLoader

    return PyGDataLoader(
        graphs,
        batch_size=min(256, len(graphs)),
        shuffle=False,
        num_workers=0,
    )


def _load_dqn_data(dataset: str, max_samples: int = 2000) -> torch.Tensor:
    """Load cached DQN state features for Q-network evaluation."""
    # Look for cached predictions from the fusion stage
    run_dir = LAKE_ROOT / dataset / "dqn_large_fusion"
    cached = run_dir / "cached_predictions.npz"
    if cached.exists():
        data = np.load(cached)
        states = data.get("states", data.get("features"))
        if states is not None:
            if len(states) > max_samples:
                states = states[:max_samples]
            return torch.tensor(states, dtype=torch.float32)

    # Fallback: generate random state vectors matching DQN input dim (15-D)
    log.warning("No cached DQN states found, generating synthetic states for landscape")
    from graphids.core.models.registry import fusion_state_dim

    dim = fusion_state_dim()
    rng = np.random.default_rng(42)
    states = rng.uniform(0, 1, size=(max_samples, dim)).astype(np.float32)
    return torch.tensor(states)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_loss_landscape(
    model_type: str,
    dataset: str,
    resolution: int = 51,
    scale: float = 1.0,
    direction_seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    """Compute 2D loss surface around a trained model minimum.

    Parameters
    ----------
    model_type : str
        One of "vgae", "gat", "dqn".
    dataset : str
        Dataset name (e.g., "hcrl_sa").
    resolution : int
        Grid resolution (resolution x resolution points).
    scale : float
        Range of perturbation in each direction [-scale, +scale].
    direction_seed : int
        Random seed for direction generation (for reproducibility).
    device : torch.device, optional
        Compute device. Defaults to CUDA if available.

    Returns
    -------
    dict
        Keys: x, y, loss (flattened arrays), plus metadata.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name, model_scale = _MODEL_RUN_MAP[model_type]
    run_dir = LAKE_ROOT / dataset / run_name
    checkpoint_file = run_dir / "best_model.pt"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    log.info("Loading model %s from %s", model_type, checkpoint_file)

    # Load config from checkpoint dir (preserves exact architecture used during training)
    from graphids.config.schema import PipelineConfig

    config_file = run_dir / "config.json"
    if config_file.exists():
        cfg = PipelineConfig.load(config_file)
        log.info("Loaded config from %s", config_file)
    else:
        cfg = resolve(f"model_type={model_type}", f"scale={model_scale}", f"dataset={dataset}")
        log.warning("No config.json in %s — using resolve() defaults", run_dir)

    # Build model and load weights
    from graphids.core.models.registry import get as get_model

    state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)

    if model_type == "dqn":
        from graphids.core.models.dqn import QNetwork

        model = QNetwork.from_config(cfg)
        # DQN checkpoint stores full agent state; extract Q-network weights
        if "q_network" in state_dict:
            state_dict = state_dict["q_network"]
    else:
        # Infer num_ids from checkpoint embedding weight shape
        num_ids = state_dict["id_embedding.weight"].shape[0]
        in_ch = NODE_FEATURE_COUNT
        log.info("Inferred num_ids=%d from checkpoint", num_ids)
        entry = get_model(model_type)
        model = entry.factory(cfg, num_ids, in_ch)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load data
    if model_type == "dqn":
        dqn_states = _load_dqn_data(dataset).to(device)
        # Wrap in a simple iterable for the DQN loss function
        dataloader = [dqn_states[i : i + 256] for i in range(0, len(dqn_states), 256)]
    else:
        graphs = _load_graph_data(dataset, model_type, cfg)
        dataloader = _make_dataloader(graphs, cfg)

    # Generate directions
    log.info("Generating filter-normalized random directions (seed=%d)", direction_seed)
    dir1 = _random_direction(model, direction_seed)
    dir2 = _random_direction(model, direction_seed + 1)

    # Save base parameters
    base_params = [p.data.clone() for p in model.parameters()]

    # Evaluate on grid
    loss_fn = _LOSS_FN[model_type]
    alphas = np.linspace(-scale, scale, resolution)
    betas = np.linspace(-scale, scale, resolution)

    log.info(
        "Computing %d x %d = %d loss evaluations (scale=%.2f)",
        resolution,
        resolution,
        resolution * resolution,
        scale,
    )

    xs, ys, losses = [], [], []
    total = resolution * resolution
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            _perturb_model(model, base_params, dir1, dir2, a, b)
            loss_val = loss_fn(model, dataloader, device)
            xs.append(float(a))
            ys.append(float(b))
            losses.append(float(loss_val))

            done = i * resolution + j + 1
            if done % max(1, total // 10) == 0:
                log.info("  Progress: %d/%d (%.0f%%)", done, total, 100 * done / total)

    # Restore base parameters
    _perturb_model(model, base_params, dir1, dir2, 0.0, 0.0)

    return {
        "x": xs,
        "y": ys,
        "loss": losses,
        "model_type": [model_type] * len(xs),
        "scale": [model_scale] * len(xs),
        "dataset": [dataset] * len(xs),
        "direction_seed": [direction_seed] * len(xs),
    }


def save_landscape(results: dict, output_dir: Path | None = None) -> Path:
    """Save loss landscape results as Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_dir is None:
        output_dir = OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    model_type = results["model_type"][0]
    dataset = results["dataset"][0]
    filename = f"{model_type}_{dataset}.parquet"

    table = pa.table(results)
    out_path = output_dir / filename
    pq.write_table(table, out_path)
    log.info("Saved loss landscape (%d points) → %s", len(results["x"]), out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="loss_landscape",
        description="Compute 2D loss landscape around trained model minimum",
    )
    parser.add_argument("--model", required=True, choices=["vgae", "gat", "dqn"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--resolution", type=int, default=51, help="Grid resolution (default: 51)")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Perturbation range (default: 1.0)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Direction seed (default: 42)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args(argv)

    from graphids.logging import configure_logging
    configure_logging()

    results = compute_loss_landscape(
        model_type=args.model,
        dataset=args.dataset,
        resolution=args.resolution,
        scale=args.scale,
        direction_seed=args.seed,
    )
    save_landscape(results, args.output_dir)


if __name__ == "__main__":
    main()
