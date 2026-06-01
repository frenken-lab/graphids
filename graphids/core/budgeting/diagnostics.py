"""Probe diagnostics for replaying failing graph model forwards."""

from __future__ import annotations

import contextlib

import torch
from structlog import get_logger

log = get_logger(__name__)


def _loss(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict):
        return out["loss"]
    return out


@contextlib.contextmanager
def _silent_log(model):
    """Silence ``self.log(...)`` during probe warmup and measurement."""
    sentinel = object()
    orig = model.__dict__.get("log", sentinel)
    model.log = lambda *a, **k: None
    try:
        yield
    finally:
        if orig is sentinel:
            del model.log
        else:
            model.log = orig


def _dump_intermediates(
    model, batch, tag: str, *, cpu_state=None, cuda_state=None, dev=None
) -> None:
    """Replay a failing forward under saved RNG state and log tensor finiteness."""
    diag: dict = {"tag": tag, "V": int(batch.num_nodes), "E": int(batch.num_edges)}
    bad_params = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad_params.append(name)
    diag["bad_params"] = bad_params

    if hasattr(model, "_forward_tensors"):
        if cpu_state is not None:
            torch.set_rng_state(cpu_state)
        if cuda_state is not None and dev is not None:
            torch.cuda.set_rng_state(cuda_state, dev)
        with torch.no_grad():
            ea = getattr(batch, "edge_attr", None)
            out = model._forward_tensors(
                batch.x, batch.edge_index, batch.batch, edge_attr=ea, node_id=batch.node_id
            )
        names = ("cont_out", "canid_logits", "nbr_pred", "z", "kl_per_node", "edge_logits")
        for name, t in zip(names, out):
            if isinstance(t, torch.Tensor):
                diag[f"{name}_has_nan"] = bool(torch.isnan(t).any().item())
                diag[f"{name}_has_inf"] = bool(torch.isinf(t).any().item())
                diag[f"{name}_finite"] = not (diag[f"{name}_has_nan"] or diag[f"{name}_has_inf"])
                diag[f"{name}_absmax"] = (
                    float(t.abs().nan_to_num(neginf=0).max().item()) if t.numel() else 0.0
                )
                diag[f"{name}_shape"] = list(t.shape)
    log.error("nan_debug_intermediates", **diag)
