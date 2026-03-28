"""Save model embeddings and attention weights as compressed NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
import torch
from graphids.core.preprocessing.datamodule import make_graph_loader

log = structlog.get_logger()


@torch.no_grad()
def collect_and_save_embeddings(
    model: torch.nn.Module,
    val_data: list,
    device: torch.device,
    output_dir: Path,
    model_type: str,
    *,
    max_samples: int = 2000,
    batch_size: int = 256,
) -> None:
    """Run inference on val_data, collect embeddings + labels, save as NPZ."""
    was_training = model.training
    model.eval()
    try:
        data = val_data[:max_samples]
        loader = make_graph_loader(data, batch_size=batch_size)

        all_emb, all_labels = [], []
        for batch in loader:
            batch = batch.clone().to(device)
            if model_type == "vgae":
                # VGAE encode() returns (z, kl_loss); z is per-node
                edge_attr = getattr(batch, "edge_attr", None)
                z, _ = model.encode(batch.x, batch.edge_index, edge_attr, batch.batch, batch.node_id)
                # Pool per-node z to per-graph via mean
                from torch_geometric.utils import scatter
                emb = scatter(z, batch.batch, dim=0, reduce="mean")
            else:
                # GATWithJK: forward(data, return_embedding=True) -> (logits, emb)
                # emb is pooled JK output before FC layers
                _, emb = model(batch, return_embedding=True)
            all_emb.append(emb.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        embeddings = np.concatenate(all_emb)
        labels = np.concatenate(all_labels)
        path = output_dir / "embeddings.npz"
        np.savez_compressed(path, embeddings=embeddings, labels=labels, model_type=model_type)
        log.info("embeddings_saved", path=str(path), n_samples=len(labels), model_type=model_type)
    finally:
        model.train(was_training)


@torch.no_grad()
def collect_and_save_attention(
    model: torch.nn.Module,
    val_data: list,
    device: torch.device,
    output_dir: Path,
    *,
    max_samples: int = 50,
    batch_size: int = 16,
) -> None:
    """Collect GAT attention weights for a subset of graphs, save as NPZ.

    Only works for GATWithJK with conv_type="gat". Logs a warning and
    returns early if the model doesn't support attention extraction.
    """
    # Guard: only GATWithJK with conv_type="gat" exposes attention weights
    conv_type = getattr(model, "conv_type", None)
    if conv_type != "gat":
        log.warning(
            "attention_extraction_skipped",
            reason=f"conv_type={conv_type!r}, only 'gat' supports return_attention_weights",
        )
        return

    was_training = model.training
    model.eval()
    try:
        data = val_data[:max_samples]
        loader = make_graph_loader(data, batch_size=batch_size)

        attn_export: dict[str, np.ndarray] = {}
        sample_idx = 0
        for batch in loader:
            batch = batch.clone().to(device)
            # forward(return_attention_weights=True) -> (xs_list, [alpha_per_layer])
            xs, attention_weights = model(batch, return_attention_weights=True)
            for i in range(batch.num_graphs):
                prefix = f"sample_{sample_idx}"
                attn_export[f"{prefix}_label"] = batch.y[i].cpu().numpy()
                for layer_idx, alpha in enumerate(attention_weights):
                    attn_export[f"{prefix}_layer_{layer_idx}_alpha"] = alpha.numpy()
                sample_idx += 1

        if attn_export:
            attn_export["n_samples"] = np.array(sample_idx)
            path = output_dir / "attention_weights.npz"
            np.savez_compressed(path, **attn_export)
            log.info("attention_weights_saved", samples=sample_idx, path=str(path))
    finally:
        model.train(was_training)
