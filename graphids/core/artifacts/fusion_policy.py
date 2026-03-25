"""Save DQN/bandit fusion policy as JSON."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from graphids.core.models.fusion_baselines import FusionResult

log = structlog.get_logger()


def save_fusion_policy(out: Path, fusion_result: FusionResult | None) -> None:
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
