"""Save VGAE and GAT embeddings + attention weights as compressed NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

from graphids.core.models.gat import GATResult
from graphids.core.models.vgae import VGAEResult

log = structlog.get_logger()


def save_embeddings(out: Path, vgae_result: VGAEResult | None, gat_result: GATResult | None) -> None:
    embed_data: dict[str, np.ndarray] = {}
    if vgae_result is not None:
        if vgae_result.embeddings is not None:
            embed_data["vgae_z"] = vgae_result.embeddings
            embed_data["vgae_labels"] = vgae_result.labels
            embed_data["vgae_errors"] = vgae_result.errors
            embed_data["vgae_attack_types"] = vgae_result.attack_types
        if vgae_result.components is not None:
            for name, arr in vgae_result.components.items():
                embed_data[f"vgae_error_{name}"] = arr
    if gat_result is not None and gat_result.embeddings is not None:
        embed_data["gat_emb"] = gat_result.embeddings
        embed_data["gat_labels"] = gat_result.labels
        embed_data["gat_attack_types"] = gat_result.attack_types
    if embed_data:
        path = out / "embeddings.npz"
        np.savez_compressed(path, **embed_data)
        log.info("embeddings_saved", path=str(path))


def save_attention(out: Path, gat_result: GATResult | None) -> None:
    if gat_result is None or not gat_result.attention:
        return
    attn_export: dict = {}
    for i, entry in enumerate(gat_result.attention):
        prefix = f"sample_{i}"
        attn_export[f"{prefix}_graph_idx"] = entry["graph_idx"]
        attn_export[f"{prefix}_label"] = entry["label"]
        attn_export[f"{prefix}_edge_index"] = entry["edge_index"]
        attn_export[f"{prefix}_node_features"] = entry["node_features"]
        for layer_idx, aw in enumerate(entry["attention_weights"]):
            attn_export[f"{prefix}_layer_{layer_idx}_alpha"] = aw
    attn_export["n_samples"] = len(gat_result.attention)
    path = out / "attention_weights.npz"
    np.savez_compressed(path, **attn_export)
    log.info("attention_weights_saved", samples=len(gat_result.attention), path=str(path))
