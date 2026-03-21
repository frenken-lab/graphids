"""Lightning callbacks and artifact save functions.

RunMetadataCallback: saves git SHA + artifact checksums after training.
save_embeddings/save_attention/save_dqn_policy: eval artifact persistence.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import structlog

from .eval_inference import FusionResult, GATResult, VGAEResult

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Eval artifact save functions (called directly, not as callbacks)
# ---------------------------------------------------------------------------

def save_embeddings(
    out: Path, vgae_result: VGAEResult | None, gat_result: GATResult | None,
) -> None:
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


def save_dqn_policy(out: Path, fusion_result: FusionResult | None) -> None:
    if fusion_result is None:
        return
    alphas = fusion_result.scores.tolist()
    labels = fusion_result.labels.tolist()
    alpha_by_label: dict[str, list] = {"normal": [], "attack": []}
    for a, lbl in zip(alphas, labels):
        alpha_by_label["normal" if lbl == 0 else "attack"].append(a)
    policy_data = {
        "alphas": alphas, "labels": labels,
        "alpha_by_label": alpha_by_label,
        "q_values": fusion_result.q_values.tolist(),
    }
    path = out / "dqn_policy.json"
    path.write_text(json.dumps(policy_data, indent=2))
    log.info("dqn_policy_saved", path=str(path))


# ---------------------------------------------------------------------------
# Lightning callbacks
# ---------------------------------------------------------------------------

class RunMetadataCallback(pl.Callback):
    """Save run metadata (git SHA, artifact checksums) after training."""

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        out = Path(trainer.log_dir)
        out.mkdir(parents=True, exist_ok=True)

        metadata: dict = {}
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            metadata["git_sha"] = sha
        except (subprocess.CalledProcessError, FileNotFoundError):
            metadata["git_sha"] = "unknown"

        checksums = {}
        for f in sorted(out.iterdir()):
            if f.is_file() and f.suffix in (".pt", ".ckpt", ".npz", ".csv", ".yaml"):
                h = hashlib.sha256()
                with open(f, "rb") as fh:
                    for chunk in iter(lambda: fh.read(8192), b""):
                        h.update(chunk)
                checksums[f.name] = h.hexdigest()
        if checksums:
            metadata["checksums"] = checksums

        path = out / "run_metadata.json"
        path.write_text(json.dumps(metadata, indent=2))
        log.info("run_metadata_saved", path=str(path))
