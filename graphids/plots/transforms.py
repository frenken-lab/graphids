"""Data loading and transforms for budget cost model. No plotting — pure math."""

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from graphids.config import PROJECT_ROOT

SAFETY_MARGIN = 0.85


class ModelParams(NamedTuple):
    """Fitted throughput model parameters for one (model_type, scale).

    Derived properties (seconds): gamma_s, alpha_train_s, beta_train_s.
    Methods: throughput, mem_budget, throughput_floor, cg_ratio,
    collation_limit, compute_limit.
    """

    gamma_us: float      # collation cost per graph (microseconds)
    alpha_ms: float      # GPU kernel overhead per step (milliseconds)
    beta_us: float       # GPU cost per node, forward-only (microseconds)
    bwd: float           # backward VRAM multiplier
    mean_nodes: float    # dataset mean nodes per graph
    bpn: int             # bytes per node (includes backward multiplier)

    @property
    def gamma_s(self) -> float: return self.gamma_us * 1e-6

    @property
    def alpha_train_s(self) -> float: return self.alpha_ms * 1e-3 * self.bwd

    @property
    def beta_train_s(self) -> float: return self.beta_us * 1e-6 * self.bwd

    def throughput(self, B: np.ndarray, num_workers: int) -> np.ndarray:
        """Throughput (nodes/sec) as function of batch size B (graphs)."""
        N = B * self.mean_nodes
        t_collate = self.gamma_s * B / num_workers
        t_gpu = self.alpha_train_s + self.beta_train_s * N
        return N / np.maximum(t_collate, t_gpu)

    def mem_budget(self, free_bytes: int) -> int:
        """VRAM ceiling: max nodes that fit with safety margin."""
        return int(free_bytes * SAFETY_MARGIN / self.bpn)

    def throughput_floor(self, num_workers: int) -> int | None:
        """Min batch size (nodes) to amortize GPU overhead alpha."""
        collation_rate = self.gamma_s / num_workers
        gpu_rate = self.beta_train_s * self.mean_nodes
        if collation_rate > gpu_rate and self.alpha_train_s > 0:
            b_floor = self.alpha_train_s / (collation_rate - gpu_rate)
            return max(1, int(math.ceil(b_floor * self.mean_nodes)))
        return None

    def cg_ratio(self, num_workers: int) -> float:
        """Collation-to-GPU ratio. >1 = collation-bound, <1 = compute-bound."""
        if self.beta_train_s <= 0:
            return float("inf")
        return (self.gamma_s / (self.mean_nodes * num_workers)) / self.beta_train_s

    def collation_limit(self, num_workers: int) -> float:
        """Asymptotic throughput (nodes/sec) when collation-bound."""
        return self.mean_nodes * num_workers / self.gamma_s

    def compute_limit(self) -> float | None:
        """Asymptotic throughput (nodes/sec) when compute-bound. None if beta=0."""
        return 1.0 / self.beta_train_s if self.beta_train_s > 0 else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_calibration_csv(path: Path) -> pl.DataFrame:
    """Read budget_calibration.csv with correct types."""
    return pl.read_csv(path)


def load_gpus(select: str | None = None) -> tuple[dict[str, int], str, int]:
    """Read GPU VRAM from configs/resources/clusters.json, optionally select one."""
    import json
    vram = json.loads(
        (PROJECT_ROOT / "configs" / "resources" / "clusters.json").read_text()
    )["gpu_vram"]
    gpus = {n.replace("_", " ").upper(): int(v["free_gb"] * 1024**3)
            for n, v in vram.items()}
    label = select.replace("_", " ").upper() if select else next(iter(gpus))
    if label not in gpus:
        raise KeyError(f"GPU '{select}' not in clusters.json. Available: {list(gpus)}")
    return gpus, label, gpus[label]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_models(df: pl.DataFrame) -> dict[str, ModelParams]:
    """Group by model_type/scale, fit throughput params via least-squares.

    gamma: median(T_collation / n_graphs). alpha, beta: linear fit of T_gpu.
    """
    models: dict[str, ModelParams] = {}
    for (mt, sc), g in df.group_by(["model_type", "scale"]):
        vc = g.filter((pl.col("n_graphs") > 0) & (pl.col("t_collation_ms") > 0))
        gammas = (vc["t_collation_ms"] * 1000 / vc["n_graphs"]).to_numpy()
        gamma_us = float(np.median(gammas)) if len(gammas) else 0.0

        vg = g.filter((pl.col("target_nodes") > 0) & (pl.col("t_gpu_ms") > 0))
        ns, ts = vg["target_nodes"].to_numpy(dtype=float), vg["t_gpu_ms"].to_numpy(dtype=float)
        if len(ns) >= 2:
            coeffs = np.polyfit(ns, ts, 1)
            alpha_ms, beta_us = max(0.0, float(coeffs[1])), float(coeffs[0]) * 1000
        elif len(ns) == 1:
            alpha_ms, beta_us = 0.0, float(ts[0]) * 1000 / float(ns[0])
        else:
            alpha_ms, beta_us = 0.0, 0.0

        first = g.row(0, named=True)
        models[f"{mt}/{sc}"] = ModelParams(
            gamma_us, alpha_ms, beta_us,
            first["backward_multiplier"], first["mean_nodes"],
            int(first["bytes_per_node"]),
        )
    return models
