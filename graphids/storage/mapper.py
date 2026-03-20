"""Translation layer. Domain-aware. Objects ↔ bytes via gateway.

Follows ZenML Materializer pattern: per-type save/load registered against
a store. Absorbs eval_writers.py artifact persistence + cache I/O.

This module DOES import from graphids.config and graphids.core (lazy, inside
functions) because it needs domain knowledge for serialization. The gateway
itself remains domain-free.
"""

from __future__ import annotations

import json
import structlog
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from .gateway import StorageGateway

log = structlog.get_logger()


class ArtifactMapper:
    """Domain-aware translation layer. Objects ↔ bytes via gateway."""

    def __init__(self, gateway: StorageGateway):
        self._gw = gateway

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, state_dict: dict, stage: str) -> Path:
        """Save a model checkpoint (best_model.pt) atomically."""
        path = self._gw.resolve(stage, "best_model.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_torch_save(state_dict, path)
        log.info("checkpoint_saved", path=str(path))
        return path

    def load_checkpoint(
        self, stage: str, model_type: str | None = None
    ) -> dict:
        """Load a model checkpoint state_dict."""
        path = self._gw.require(stage, "best_model.pt", model_type=model_type)
        return torch.load(path, map_location="cpu", weights_only=True)

    def save_dqn_checkpoint(self, agent_state: dict, stage: str) -> Path:
        """Save DQN agent checkpoint (q_network + target_network + epsilon)."""
        path = self._gw.resolve(stage, "best_model.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_torch_save(agent_state, path)
        log.info("dqn_checkpoint_saved", path=str(path))
        return path

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def save_config(self, cfg, stage: str) -> Path:
        """Save frozen PipelineConfig as JSON."""
        path = self._gw.resolve(stage, "config.json")
        self._gw.write_json(path, cfg.model_dump())
        log.info("config_saved", path=str(path))
        return path

    def load_config(self, stage: str, model_type: str | None = None):
        """Load frozen PipelineConfig from JSON.

        Returns a PipelineConfig instance.
        """
        from graphids.config import PipelineConfig

        path = self._gw.require(stage, "config.json", model_type=model_type)
        raw = self._gw.read_json(path)
        return PipelineConfig.model_validate(raw)

    # ------------------------------------------------------------------
    # Training result (combined save)
    # ------------------------------------------------------------------

    def save_training_result(self, model, cfg, stage: str, trainer) -> dict:
        """Save checkpoint + config. Returns result dict with path and metrics.

        Metrics extraction is non-fatal — returns empty metrics on failure.
        """
        import pytorch_lightning as pl

        try:
            metrics = _extract_training_metrics(trainer)
        except Exception as e:
            log.warning("training_metrics_extraction_failed", error=str(e))
            metrics = {}

        ckpt_path = self.save_checkpoint(model.state_dict(), stage)
        self.save_config(cfg, stage)
        return {"checkpoint": str(ckpt_path), "metrics": metrics}

    # ------------------------------------------------------------------
    # Evaluation artifacts (absorbed from eval_writers.py)
    # ------------------------------------------------------------------

    def save_embeddings(self, gat, vgae, stage: str) -> None:
        """Write GAT and VGAE embeddings to embeddings.npz."""
        embed_data: dict[str, np.ndarray] = {}

        if vgae is not None:
            if vgae.embeddings is not None:
                embed_data["vgae_z"] = vgae.embeddings
                embed_data["vgae_labels"] = vgae.labels
                embed_data["vgae_errors"] = vgae.errors
                embed_data["vgae_attack_types"] = vgae.attack_types
            if vgae.components is not None:
                for comp_name, comp_arr in vgae.components.items():
                    embed_data[f"vgae_error_{comp_name}"] = comp_arr

        if gat is not None and gat.embeddings is not None:
            embed_data["gat_emb"] = gat.embeddings
            embed_data["gat_labels"] = gat.labels
            embed_data["gat_attack_types"] = gat.attack_types

        if embed_data:
            npz_path = self._gw.resolve(stage, "embeddings.npz")
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, **embed_data)
            log.info("embeddings_saved", path=str(npz_path))

    def save_attention(self, gat, stage: str) -> None:
        """Write GAT attention weights to attention_weights.npz."""
        if gat is None or not gat.attention:
            return

        attn_export: dict = {}
        for i, entry in enumerate(gat.attention):
            prefix = f"sample_{i}"
            attn_export[f"{prefix}_graph_idx"] = entry["graph_idx"]
            attn_export[f"{prefix}_label"] = entry["label"]
            attn_export[f"{prefix}_edge_index"] = entry["edge_index"]
            attn_export[f"{prefix}_node_features"] = entry["node_features"]
            for layer_idx, aw in enumerate(entry["attention_weights"]):
                attn_export[f"{prefix}_layer_{layer_idx}_alpha"] = aw
        attn_export["n_samples"] = len(gat.attention)

        attn_path = self._gw.resolve(stage, "attention_weights.npz")
        attn_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(attn_path, **attn_export)
        log.info("attention_weights_saved", samples=len(gat.attention), path=str(attn_path))

    def save_dqn_policy(self, fusion, stage: str) -> None:
        """Write DQN policy data (alphas, q-values) to dqn_policy.json."""
        if fusion is None:
            return

        alphas = fusion.scores.tolist()
        labels = fusion.labels.tolist()

        alpha_by_label: dict[str, list] = {"normal": [], "attack": []}
        for a, lbl in zip(alphas, labels):
            alpha_by_label["normal" if lbl == 0 else "attack"].append(a)

        policy_data = {
            "alphas": alphas,
            "labels": labels,
            "alpha_by_label": alpha_by_label,
            "q_values": fusion.q_values.tolist(),
        }
        path = self._gw.resolve(stage, "dqn_policy.json")
        self._gw.write_json(path, policy_data)

    def save_cka(
        self,
        cfg,
        val_data,
        device,
        num_ids: int,
        in_ch: int,
        stage: str,
    ) -> None:
        """Compute and save CKA matrix between teacher and student GAT layers."""
        from graphids.config import resolve

        teacher_cfg = resolve("gat", "large", dataset=cfg.dataset)
        if not self._gw.exists("curriculum", "best_model.pt", model_type="gat"):
            log.warning("CKA: teacher checkpoint not found")
            return
        # Check teacher with teacher_cfg's gateway
        teacher_gw = StorageGateway(cfg=teacher_cfg)
        if not teacher_gw.exists("curriculum", "best_model.pt"):
            log.warning("CKA: teacher checkpoint not found")
            return

        if not self._gw.exists("curriculum", "best_model.pt", model_type="gat"):
            log.warning("CKA: student checkpoint not found")
            return

        from graphids.pipeline.stages.data_loading import cleanup
        from graphids.pipeline.stages.trainer_factory import load_model

        teacher = load_model(teacher_cfg, "gat", "curriculum", num_ids, in_ch, device)
        student = load_model(cfg, "gat", "curriculum", num_ids, in_ch, device)

        from graphids.pipeline.stages.cka import _collect_layer_representations, _linear_cka

        teacher_layers = _collect_layer_representations(teacher, val_data, device)
        student_layers = _collect_layer_representations(student, val_data, device)

        if not teacher_layers or not student_layers:
            log.warning("CKA: empty layer representations")
            return

        n_teacher = len(teacher_layers)
        n_student = len(student_layers)
        cka_matrix = np.zeros((n_teacher, n_student))
        for i in range(n_teacher):
            for j in range(n_student):
                cka_matrix[i, j] = _linear_cka(teacher_layers[i], student_layers[j])

        cka_data = {
            "matrix": cka_matrix.tolist(),
            "teacher_layers": [f"Teacher L{i + 1}" for i in range(n_teacher)],
            "student_layers": [f"Student L{i + 1}" for i in range(n_student)],
        }
        path = self._gw.resolve(stage, "cka_matrix.json")
        self._gw.write_json(path, cka_data)
        log.info("cka_matrix_saved", teacher_layers=n_teacher, student_layers=n_student)

        del teacher, student
        cleanup()

    # ------------------------------------------------------------------
    # Cache I/O (absorbed from _atomic_io.py)
    # ------------------------------------------------------------------

    def save_collated(self, graphs: list, cache_path: Path) -> None:
        """Save collated graphs atomically: write to tmp, fsync, rename."""
        from graphids.core.preprocessing._dataset import save_collated

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        try:
            save_collated(graphs, tmp_path)
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
            from .gateway import _atomic_rename

            _atomic_rename(tmp_path, cache_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def load_collated(self, cache_path: Path):
        """Load collated graphs from cache file."""
        from graphids.core.preprocessing._dataset import load_collated

        return load_collated(cache_path)

    # ------------------------------------------------------------------
    # Pickle (absorbs vocabulary save/load patterns)
    # ------------------------------------------------------------------

    def save_pickle(self, obj, path: Path) -> None:
        """Save object via pickle atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(obj, f, protocol=4)
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
            from .gateway import _atomic_rename

            _atomic_rename(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def load_pickle(self, path: Path) -> object:
        """Load object from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301

    # ------------------------------------------------------------------
    # Generic
    # ------------------------------------------------------------------

    def save_npz(self, data: dict, stage: str, name: str) -> Path:
        """Save dict of arrays as compressed .npz."""
        path = self._gw.resolve(stage, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
        log.info("npz_saved", path=str(path))
        return path

    def save_json(self, data: dict, stage: str, name: str) -> Path:
        """Save dict as JSON to stage directory."""
        path = self._gw.resolve(stage, name)
        self._gw.write_json(path, data)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _atomic_torch_save(self, obj, path: Path) -> None:
        """Save a torch object atomically via tmpfile + fsync + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        try:
            torch.save(obj, tmp_path)
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
            from .gateway import _atomic_rename

            _atomic_rename(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def open_gateway(cfg) -> tuple[StorageGateway, ArtifactMapper]:
    """Create a gateway + mapper pair from a PipelineConfig."""
    gw = StorageGateway(cfg=cfg)
    return gw, ArtifactMapper(gw)


# ---------------------------------------------------------------------------
# Training metrics extraction (absorbed from training.py)
# ---------------------------------------------------------------------------


def _extract_training_metrics(trainer) -> dict:
    """Extract metrics from trainer's callback state after training."""
    import pytorch_lightning as pl

    metrics: dict = {}
    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.ModelCheckpoint) and cb.best_model_score is not None:
            metrics["val_loss"] = float(cb.best_model_score)
            break

    if trainer.callback_metrics:
        for k, v in trainer.callback_metrics.items():
            if k not in metrics:
                try:
                    metrics[k] = float(v) if hasattr(v, "item") else v
                except (TypeError, ValueError):
                    pass
    metrics["epochs_run"] = trainer.current_epoch + 1
    return metrics
