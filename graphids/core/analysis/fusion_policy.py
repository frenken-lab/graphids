"""Extract and save DQN/bandit fusion policy as JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from graphids.log import get_logger

log = get_logger(__name__)


def run_fusion_policy(
    *,
    module: torch.nn.Module,
    dataset: str,
    lake_root: str,
    seed: int,
    vgae_ckpt_path: str,
    gat_ckpt_path: str,
    window_size: int,
    stride: int,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Build fusion data, run agent prediction, save policy JSON."""
    from graphids.core.data.datamodule.fusion import FusionDataModule

    log.info("artifact_start", artifact="fusion_policy")
    agent = module.agent
    dm = FusionDataModule(
        dataset=dataset,
        lake_root=lake_root,
        seed=seed,
        vgae_ckpt_path=vgae_ckpt_path,
        gat_ckpt_path=gat_ckpt_path,
        window_size=window_size,
        stride=stride,
    )
    dm.setup("test")
    states = dm.val_cache["states"].to(device)
    labels = dm.val_cache["labels"]
    result = agent.predict(states)
    _save_fusion_policy(
        output_dir,
        alphas=result["alphas"].cpu().numpy(),
        labels=labels.numpy(),
        q_values=agent.q_values(result["norm_states"]).cpu().numpy(),
    )


def _save_fusion_policy(
    out: Path,
    alphas: np.ndarray,
    labels: np.ndarray,
    q_values: np.ndarray,
) -> None:
    """Serialize fusion policy data to JSON."""
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
