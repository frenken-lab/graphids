"""Budget configuration resolved from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

MB = 1024 * 1024


@dataclass(frozen=True)
class BudgetConfig:
    safety_margin: float = 0.85
    edge_headroom: float = 1.1
    probe_seed: int = 20260506
    cudnn_reserve: float = 0.05
    heuristic_bytes_per_node: int = 256 * 1024
    heuristic_bytes_per_edge: int = 32 * 1024
    heuristic_gps_bytes_per_node2: float = 32768
    default_target_bytes: int = 8 * 1024**3
    default_edges_per_node: float = 4.0
    mode: str = "auto"
    strict_probe: bool = False

    @classmethod
    def from_env(cls) -> BudgetConfig:
        return cls(
            safety_margin=float(os.environ.get("GRAPHIDS_BUDGET_SAFETY_MARGIN", "0.85")),
            edge_headroom=float(os.environ.get("GRAPHIDS_EMPIRICAL_EPN_HEADROOM", "1.1")),
            probe_seed=int(os.environ.get("GRAPHIDS_PROBE_SEED", "20260506")),
            cudnn_reserve=float(os.environ.get("GRAPHIDS_BUDGET_CUDNN_RESERVE", "0.05")),
            heuristic_bytes_per_node=int(
                os.environ.get("GRAPHIDS_BUDGET_BYTES_PER_NODE", str(256 * 1024))
            ),
            heuristic_bytes_per_edge=int(
                os.environ.get("GRAPHIDS_BUDGET_BYTES_PER_EDGE", str(32 * 1024))
            ),
            heuristic_gps_bytes_per_node2=float(
                os.environ.get("GRAPHIDS_BUDGET_GPS_BYTES_PER_NODE2", "32768")
            ),
            default_target_bytes=int(
                os.environ.get("GRAPHIDS_BUDGET_DEFAULT_TARGET_BYTES", str(8 * 1024**3))
            ),
            default_edges_per_node=float(os.environ.get("GRAPHIDS_BUDGET_EDGES_PER_NODE", "4.0")),
            mode=os.environ.get("GRAPHIDS_BUDGET_MODE", "auto").lower(),
            strict_probe=os.environ.get("GRAPHIDS_BUDGET_STRICT_PROBE", "0") == "1",
        )
