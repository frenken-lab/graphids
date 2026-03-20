"""Lightning callbacks for experiment artifact persistence.

EvalArtifactCallback: saves embeddings, attention, DQN policy after test.
RunMetadataCallback: saves git SHA + artifact checksums after training.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import structlog

from .eval_types import FusionResult, GATResult, VGAEResult

log = structlog.get_logger()


class EvalArtifactCallback(pl.Callback):
    """Save eval artifacts (embeddings, attention, DQN policy) to trainer.log_dir."""

    def __init__(self):
        self.gat_result: GATResult | None = None
        self.vgae_result: VGAEResult | None = None
        self.fusion_result: FusionResult | None = None

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        out = Path(trainer.log_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._save_embeddings(out)
        self._save_attention(out)
        self._save_dqn_policy(out)

    def _save_embeddings(self, out: Path) -> None:
        embed_data: dict[str, np.ndarray] = {}

        if self.vgae_result is not None:
            if self.vgae_result.embeddings is not None:
                embed_data["vgae_z"] = self.vgae_result.embeddings
                embed_data["vgae_labels"] = self.vgae_result.labels
                embed_data["vgae_errors"] = self.vgae_result.errors
                embed_data["vgae_attack_types"] = self.vgae_result.attack_types
            if self.vgae_result.components is not None:
                for comp_name, comp_arr in self.vgae_result.components.items():
                    embed_data[f"vgae_error_{comp_name}"] = comp_arr

        if self.gat_result is not None and self.gat_result.embeddings is not None:
            embed_data["gat_emb"] = self.gat_result.embeddings
            embed_data["gat_labels"] = self.gat_result.labels
            embed_data["gat_attack_types"] = self.gat_result.attack_types

        if embed_data:
            path = out / "embeddings.npz"
            np.savez_compressed(path, **embed_data)
            log.info("embeddings_saved", path=str(path))

    def _save_attention(self, out: Path) -> None:
        if self.gat_result is None or not self.gat_result.attention:
            return

        attn_export: dict = {}
        for i, entry in enumerate(self.gat_result.attention):
            prefix = f"sample_{i}"
            attn_export[f"{prefix}_graph_idx"] = entry["graph_idx"]
            attn_export[f"{prefix}_label"] = entry["label"]
            attn_export[f"{prefix}_edge_index"] = entry["edge_index"]
            attn_export[f"{prefix}_node_features"] = entry["node_features"]
            for layer_idx, aw in enumerate(entry["attention_weights"]):
                attn_export[f"{prefix}_layer_{layer_idx}_alpha"] = aw
        attn_export["n_samples"] = len(self.gat_result.attention)

        path = out / "attention_weights.npz"
        np.savez_compressed(path, **attn_export)
        log.info("attention_weights_saved", samples=len(self.gat_result.attention), path=str(path))

    def _save_dqn_policy(self, out: Path) -> None:
        if self.fusion_result is None:
            return

        alphas = self.fusion_result.scores.tolist()
        labels = self.fusion_result.labels.tolist()

        alpha_by_label: dict[str, list] = {"normal": [], "attack": []}
        for a, lbl in zip(alphas, labels):
            alpha_by_label["normal" if lbl == 0 else "attack"].append(a)

        policy_data = {
            "alphas": alphas,
            "labels": labels,
            "alpha_by_label": alpha_by_label,
            "q_values": self.fusion_result.q_values.tolist(),
        }
        path = out / "dqn_policy.json"
        path.write_text(json.dumps(policy_data, indent=2))
        log.info("dqn_policy_saved", path=str(path))


class RunMetadataCallback(pl.Callback):
    """Save run metadata (git SHA, artifact checksums) after training."""

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        out = Path(trainer.log_dir)
        out.mkdir(parents=True, exist_ok=True)

        metadata: dict = {}

        # Git SHA
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            metadata["git_sha"] = sha
        except (subprocess.CalledProcessError, FileNotFoundError):
            metadata["git_sha"] = "unknown"

        # Checksums of artifacts in output dir
        checksums = {}
        for f in sorted(out.iterdir()):
            if f.is_file() and f.suffix in (".pt", ".ckpt", ".npz", ".csv", ".yaml"):
                checksums[f.name] = _sha256(f)
        if checksums:
            metadata["checksums"] = checksums

        path = out / "run_metadata.json"
        path.write_text(json.dumps(metadata, indent=2))
        log.info("run_metadata_saved", path=str(path))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
