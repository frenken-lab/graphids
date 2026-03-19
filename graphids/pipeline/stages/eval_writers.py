"""Artifact writers for evaluation stage.

Each function accepts typed results and writes to disk. CKA is self-contained
(loads models, computes, writes).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from graphids.config import PipelineConfig

from .eval_types import FusionResult, GATResult, VGAEResult

log = logging.getLogger(__name__)


def write_embeddings(gat: GATResult | None, vgae: VGAEResult | None, out_dir: Path) -> None:
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
        npz_path = out_dir / "embeddings.npz"
        np.savez_compressed(npz_path, **embed_data)
        log.info("Saved embeddings → %s", npz_path)


def write_attention(gat: GATResult | None, out_dir: Path) -> None:
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

    attn_path = out_dir / "attention_weights.npz"
    np.savez_compressed(attn_path, **attn_export)
    log.info("Saved attention weights (%d samples) → %s", len(gat.attention), attn_path)


def write_dqn_policy(fusion: FusionResult | None, out_dir: Path) -> None:
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
    policy_path = out_dir / "dqn_policy.json"
    policy_path.write_text(json.dumps(policy_data, indent=2))
    log.info("Saved DQN policy → %s", policy_path)


def write_cka(
    cfg: PipelineConfig,
    val_data,
    device,
    num_ids: int,
    in_ch: int,
    out_dir: Path,
) -> None:
    """Compute and save CKA matrix between teacher and student GAT layers."""
    from graphids.config import resolve
    from graphids.pipeline.artifacts import artifact_exists

    from .utils import cleanup, load_model

    teacher_cfg = resolve("gat", "large", dataset=cfg.dataset)
    if not artifact_exists(teacher_cfg, "curriculum", "best_model.pt", model_type="gat"):
        log.warning("CKA: teacher checkpoint not found")
        return

    if not artifact_exists(cfg, "curriculum", "best_model.pt", model_type="gat"):
        log.warning("CKA: student checkpoint not found")
        return

    teacher = load_model(teacher_cfg, "gat", "curriculum", num_ids, in_ch, device)
    student = load_model(cfg, "gat", "curriculum", num_ids, in_ch, device)

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
    cka_path = out_dir / "cka_matrix.json"
    cka_path.write_text(json.dumps(cka_data, indent=2))
    log.info("Saved CKA matrix (%dx%d) → %s", n_teacher, n_student, cka_path)

    del teacher, student
    cleanup()


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear CKA between two representation matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    n = X.shape[0]

    hsic_xy = np.linalg.norm(Y.T @ X, "fro") ** 2 / (n - 1) ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2 / (n - 1) ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2 / (n - 1) ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


def _collect_layer_representations(model, data, device, max_samples: int = 500) -> list[np.ndarray]:
    """Collect per-layer representations from a GAT model."""
    all_layers: list[list] | None = None
    count = 0
    with torch.no_grad():
        for g in data:
            if count >= max_samples:
                break
            g = g.clone().to(device)
            xs = model(g, return_intermediate=True)
            layer_reps = [x.mean(dim=0).cpu().numpy() for x in xs]
            if all_layers is None:
                all_layers = [[] for _ in range(len(layer_reps))]
            for i, rep in enumerate(layer_reps):
                all_layers[i].append(rep)
            count += 1
    if all_layers is None:
        return []
    return [np.array(layer) for layer in all_layers]
