"""Pure compute primitives — no filesystem, no MLflow, no logging side-effects.

Each ``compute_*`` returns a frozen dataclass (or plain dict, for CKA's
single layer→score mapping) that ``io.save_*`` knows how to serialize.
The analyzer wraps the whole batch in :func:`eval_mode`, so no compute
function re-enters it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import scatter

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EmbeddingsResult:
    embeddings: np.ndarray
    labels: np.ndarray
    model_type: str


@dataclass(frozen=True, eq=False)
class AttentionResult:
    weights: dict[str, np.ndarray]  # sample_i_layer_j_alpha + sample_i_label
    n_samples: int


@dataclass(frozen=True, eq=False)
class LandscapeResult:
    x: list[float]
    y: list[float]
    loss: list[float]
    model_type: str
    dataset: str


@dataclass(frozen=True, eq=False)
class PolicyResult:
    alphas: np.ndarray
    labels: np.ndarray
    q_values: np.ndarray


# ---------------------------------------------------------------------------
# Embeddings + attention
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_embeddings(
    model: torch.nn.Module,
    val_data: list,
    device: torch.device,
    *,
    model_type: str,
    max_samples: int = 2000,
    batch_size: int = 256,
) -> EmbeddingsResult:
    """Pool per-graph embeddings + labels for ``val_data[:max_samples]``."""
    loader = PyGDataLoader(val_data[:max_samples], batch_size=batch_size)
    all_emb, all_labels = [], []
    for batch in loader:
        batch = batch.clone().to(device)
        edge_attr = getattr(batch, "edge_attr", None)
        if model_type == "vgae":
            z, _ = model.encode(batch.x, batch.edge_index, edge_attr, batch.batch, batch.node_id)
            emb = scatter(z, batch.batch, dim=0, reduce="mean")
        elif model_type == "dgi":
            z = model.encode(batch.x, batch.edge_index, edge_attr, batch.batch, batch.node_id)
            emb = scatter(z, batch.batch, dim=0, reduce="mean")
        else:
            # GATWithJK: forward(data, return_embedding=True) → (logits, emb_pooled)
            _, emb = model(batch, return_embedding=True)
        all_emb.append(emb.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    return EmbeddingsResult(
        embeddings=np.concatenate(all_emb),
        labels=np.concatenate(all_labels),
        model_type=model_type,
    )


@torch.no_grad()
def compute_attention(
    model: torch.nn.Module,
    val_data: list,
    device: torch.device,
    *,
    max_samples: int = 50,
) -> AttentionResult | None:
    """Per-sample per-layer GAT attention weights. ``None`` if model lacks them."""
    if getattr(model, "conv_type", None) != "gat":
        return None

    loader = PyGDataLoader(val_data[:max_samples], batch_size=1)
    out: dict[str, np.ndarray] = {}
    sample_idx = 0
    for batch in loader:
        batch = batch.clone().to(device)
        _xs, attention_weights = model(batch, return_attention_weights=True)
        prefix = f"sample_{sample_idx}"
        out[f"{prefix}_label"] = batch.y[0].cpu().numpy()
        for layer_idx, alpha in enumerate(attention_weights):
            out[f"{prefix}_layer_{layer_idx}_alpha"] = alpha.cpu().numpy()
        sample_idx += 1
    return AttentionResult(weights=out, n_samples=sample_idx)


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------


def _unbiased_hsic(K: torch.Tensor, L: torch.Tensor) -> float:
    n = K.shape[0]
    ones = torch.ones(n, 1, device=K.device)
    result = torch.trace(K @ L)
    result += ((ones.T @ K @ ones @ ones.T @ L @ ones) / ((n - 1) * (n - 2))).item()
    result -= ((ones.T @ K @ L @ ones) * 2 / (n - 2)).item()
    return (result / (n * (n - 3))).item()


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X_t = torch.from_numpy(X - X.mean(axis=0)).float()
    Y_t = torch.from_numpy(Y - Y.mean(axis=0)).float()
    K, L = X_t @ X_t.T, Y_t @ Y_t.T
    denom = (_unbiased_hsic(K, K) * _unbiased_hsic(L, L)) ** 0.5
    return _unbiased_hsic(K, L) / denom if denom > 0 else 0.0


def _collect_reps(
    model: torch.nn.Module, data: list, device: torch.device, max_samples: int
) -> list[np.ndarray]:
    layers: list[list] | None = None
    count = 0
    with torch.no_grad():
        for g in data:
            if count >= max_samples:
                break
            g = g.clone().to(device, non_blocking=True)
            xs = model(g, return_intermediate=True)
            reps = [x.mean(dim=0).cpu().numpy() for x in xs]
            if layers is None:
                layers = [[] for _ in reps]
            for i, r in enumerate(reps):
                layers[i].append(r)
            count += 1
    return [np.array(l) for l in layers] if layers else []


def compute_cka(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    val_data: list,
    device: torch.device,
    *,
    max_samples: int = 500,
) -> dict[str, float]:
    """Layer-wise linear CKA between teacher and student over ``val_data``."""
    student_reps = _collect_reps(student, val_data, device, max_samples)
    teacher_reps = _collect_reps(teacher, val_data, device, max_samples)
    n_layers = min(len(teacher_reps), len(student_reps))
    return {f"layer_{i}": _linear_cka(teacher_reps[i], student_reps[i]) for i in range(n_layers)}


# ---------------------------------------------------------------------------
# Loss landscape (Li et al., 2018)
# ---------------------------------------------------------------------------


def _filter_normalize(
    direction: list[torch.Tensor], reference: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Per-filter (2D+) or global (1D) norm-match each direction tensor to its parameter."""
    out = []
    for d, r in zip(direction, reference):
        if d.dim() >= 2:
            d_flat = d.reshape(d.shape[0], -1)
            r_flat = r.reshape(r.shape[0], -1)
            r_norms = r_flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
            d_norms = d_flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
            out.append((d_flat * (r_norms / d_norms)).reshape(d.shape))
        else:
            r_norm = r.norm().clamp(min=1e-10)
            d_norm = d.norm().clamp(min=1e-10)
            out.append(d * (r_norm / d_norm))
    return out


def _random_direction(model: torch.nn.Module, seed: int) -> list[torch.Tensor]:
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
    for p, b, d1, d2 in zip(model.parameters(), base_params, dir1, dir2):
        p.data.copy_(b + alpha * d1 + beta * d2)


@torch.no_grad()
def _vgae_loss(model, dataloader, device: torch.device, cfg) -> float:
    """VGAE reconstruction + KL. Unmasked — landscape visualizes geometry around
    trained weights; random masking would inject noise. ``kl_weight`` falls back
    to the loss-config default (0.01) when not on ``cfg`` (loss-fn weights live
    on the loss module, not on module ``hparams``).
    """
    kl_weight = float(getattr(cfg, "kl_weight", 0.01))
    total, count = 0.0, 0
    for batch in dataloader:
        batch = batch.clone().to(device)
        edge_attr = getattr(batch, "edge_attr", None)
        cont, _z, kl_per_node = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr, node_id=batch.node_id
        )
        recon = F.mse_loss(cont, batch.x)
        loss = recon + kl_weight * kl_per_node.mean()
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return total / max(count, 1)


@torch.no_grad()
def _gat_loss(model, dataloader, device: torch.device, _cfg) -> float:
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
    total, count = 0.0, 0
    for batch in dataloader:
        batch = batch.clone().to(device)
        edge_attr = getattr(batch, "edge_attr", None)
        pos_z, neg_z, summary = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr, node_id=batch.node_id
        )
        loss = model.dgi_loss(pos_z, neg_z, summary, batch.batch)
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return total / max(count, 1)


_LOSS_FN = {"vgae": _vgae_loss, "gat": _gat_loss, "dgi": _dgi_loss}


def compute_landscape(
    model: torch.nn.Module,
    model_type: str,
    val_data: list,
    device: torch.device,
    hparams,
    *,
    resolution: int = 51,
    scale: float = 1.0,
    seed: int = 42,
    max_graphs: int = 500,
    dataset: str = "",
) -> LandscapeResult:
    """Loss on a ``resolution × resolution`` grid of filter-normalized perturbations.

    ``KeyError`` on unknown ``model_type`` — dispatch's ``applies_to`` should
    filter callers; reaching this with an unsupported type is a routing bug.
    """
    loss_fn = _LOSS_FN[model_type]
    if len(val_data) > max_graphs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(val_data), max_graphs, replace=False)
        data = [val_data[i] for i in idx]
    else:
        data = val_data
    dataloader = PyGDataLoader(data, batch_size=min(256, len(data)))

    dir1 = _random_direction(model, seed)
    dir2 = _random_direction(model, seed + 1)
    base = [p.data.clone() for p in model.parameters()]
    alphas = np.linspace(-scale, scale, resolution)
    betas = np.linspace(-scale, scale, resolution)

    xs, ys, losses = [], [], []
    for a in alphas:
        for b in betas:
            _perturb_model(model, base, dir1, dir2, a, b)
            losses.append(loss_fn(model, dataloader, device, hparams))
            xs.append(float(a))
            ys.append(float(b))
    _perturb_model(model, base, dir1, dir2, 0.0, 0.0)  # restore

    return LandscapeResult(x=xs, y=ys, loss=losses, model_type=model_type, dataset=dataset)


# ---------------------------------------------------------------------------
# Fusion policy
# ---------------------------------------------------------------------------


def compute_fusion_policy(
    agent, states: torch.Tensor, labels: torch.Tensor
) -> PolicyResult:
    """Run the agent on pre-built fusion states; return alphas + Q-values + labels."""
    result = agent.predict(states)
    return PolicyResult(
        alphas=result["alphas"].cpu().numpy(),
        labels=labels.cpu().numpy(),
        q_values=agent.q_values(result["norm_states"]).cpu().numpy(),
    )
