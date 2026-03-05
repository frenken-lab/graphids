"""GNNExplainer integration for feature importance analysis.

Wraps trained GAT/VGAE models for use with PyG's Explainer API,
producing per-node and per-edge importance masks that reveal
*which features* drive predictions (complementing attention weights
which show *which edges* the model attends to).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from graphids.config.constants import get_batch_index

log = logging.getLogger(__name__)


class _GATExplainerWrapper(nn.Module):
    """Adapts GATWithJK's forward(data) to Explainer's (x, edge_index, batch, edge_attr) convention."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        from torch_geometric.data import Data

        data = Data(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        return self.model(data)


class _VGAEExplainerWrapper(nn.Module):
    """Adapts VGAE's forward to return 2-class logits from reconstruction error."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        import torch.nn.functional as F

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        cont, _, _, _, _ = self.model(x, edge_index, batch, edge_attr=edge_attr)
        # Reconstruction error per node, then mean-pool to graph level
        node_error = F.mse_loss(cont, x[:, 1:], reduction="none").mean(dim=1)
        from torch_geometric.nn import global_mean_pool

        graph_error = global_mean_pool(node_error.unsqueeze(1), batch).squeeze(1)
        # Convert to 2-class logits: [low_error, high_error]
        logits = torch.stack([-graph_error, graph_error], dim=1)
        return logits


def _wrap_for_explainer(model: nn.Module, model_type: str) -> nn.Module:
    """Wrap a model for PyG Explainer compatibility."""
    if model_type == "gat":
        return _GATExplainerWrapper(model)
    elif model_type == "vgae":
        return _VGAEExplainerWrapper(model)
    else:
        raise ValueError(f"Unsupported model_type for explainer: {model_type}")


def explain_graphs(
    model: nn.Module,
    model_type: str,
    graphs: list,
    device: torch.device,
    n_samples: int = 50,
    epochs: int = 200,
) -> dict:
    """Run GNNExplainer on sampled graphs, return importance masks.

    Args:
        model: Trained model (GATWithJK or GraphAutoencoderNeighborhood).
        model_type: "gat" or "vgae".
        graphs: List of PyG Data objects.
        device: Torch device.
        n_samples: Number of graphs to explain.
        epochs: Training epochs for GNNExplainer per graph.

    Returns:
        Dict with numpy arrays: node_masks, edge_masks, graph_indices,
        labels, predictions.
    """
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.explain.config import (
        ExplanationType,
        ModelMode,
        ModelTaskLevel,
    )

    wrapped = _wrap_for_explainer(model, model_type)
    wrapped.eval()

    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type=ExplanationType.model,
        model_config=dict(
            mode=ModelMode.multiclass_classification,
            task_level=ModelTaskLevel.graph,
            return_type="raw",
        ),
        node_mask_type="attributes",
        edge_mask_type="object",
    )

    sample_indices = list(range(min(n_samples, len(graphs))))
    node_masks, edge_masks = [], []
    graph_indices, labels, predictions = [], [], []

    for idx in sample_indices:
        g = graphs[idx].clone().to(device)
        batch = get_batch_index(g, device)
        edge_attr = getattr(g, "edge_attr", None)

        try:
            explanation = explainer(
                g.x,
                g.edge_index,
                batch=batch,
                edge_attr=edge_attr,
            )
            nm = (
                explanation.node_mask.detach().cpu().numpy()
                if explanation.node_mask is not None
                else None
            )
            em = (
                explanation.edge_mask.detach().cpu().numpy()
                if explanation.edge_mask is not None
                else None
            )

            node_masks.append(nm)
            edge_masks.append(em)
            graph_indices.append(idx)

            label = g.y.item() if g.y.dim() == 0 else int(g.y[0].item())
            labels.append(label)

            with torch.no_grad():
                logits = wrapped(g.x, g.edge_index, batch=batch, edge_attr=edge_attr)
                pred = logits.argmax(dim=1)[0].item()
            predictions.append(pred)

        except Exception as e:
            log.warning("Explainer failed on graph %d: %s", idx, e)
            continue

    result = {
        "graph_indices": np.array(graph_indices),
        "labels": np.array(labels),
        "predictions": np.array(predictions),
    }

    # Store masks as object arrays (variable sizes per graph)
    if node_masks:
        result["node_masks"] = np.array(node_masks, dtype=object)
    if edge_masks:
        result["edge_masks"] = np.array(edge_masks, dtype=object)

    log.info("Explained %d/%d graphs", len(graph_indices), len(sample_indices))
    return result
