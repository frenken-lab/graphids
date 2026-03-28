"""Save DQN/bandit fusion policy as JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger()


def save_fusion_policy(
    out: Path,
    alphas: np.ndarray,
    labels: np.ndarray,
    q_values: np.ndarray,
) -> None:
    """Save fusion policy data (alpha weights, labels, Q-values) as JSON."""
    alpha_list = alphas.tolist()
    label_list = labels.tolist()
    alpha_by_label: dict[str, list] = {"normal": [], "attack": []}
    for a, lbl in zip(alpha_list, label_list):
        alpha_by_label["normal" if lbl == 0 else "attack"].append(a)
    policy_data = {
        "alphas": alpha_list,
        "labels": label_list,
        "alpha_by_label": alpha_by_label,
        "q_values": q_values.tolist(),
    }
    path = out / "dqn_policy.json"
    path.write_text(json.dumps(policy_data, indent=2))
    log.info("dqn_policy_saved", path=str(path), n_samples=len(label_list))
