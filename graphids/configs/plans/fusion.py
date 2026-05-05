"""Fusion plan.

Emits one extract row + 4 method-specific (fit, test) pairs. Submit:
extract first, then chain method rows via ``--depends-on-afterok``.
"""

from __future__ import annotations

from typing import Any

from graphids.config.catalog import best_ckpt, states_dir
from graphids.configs.compose import fusion
from graphids.configs.lib import BANDIT, DQN, MLP_FUSION, REWARD, WAVG_FUSION, spec
from graphids.configs.row import extract


def build(*, dataset: str, seed: int) -> list[dict[str, Any]]:
    extract_dir = states_dir(dataset, seed)
    vgae_ckpt = best_ckpt(dataset, "unsupervised", "vgae", seed)
    gat_ckpt = best_ckpt(dataset, "gat_loss", "focal", seed)

    def meta(variant: str) -> dict[str, Any]:
        return {
            "group": "fusion",
            "variant": variant,
            "dataset": dataset,
            "seed": seed,
            "model_type": "fusion",
            "scale": "small",
        }

    def fuse(variant: str, model: dict[str, Any]):
        return fusion(model=model, method=variant, meta=meta(variant))

    bandit = fuse("bandit", spec(BANDIT, reward_kwargs=dict(REWARD)))
    dqn = fuse("dqn", spec(DQN, reward_kwargs=dict(REWARD)))
    mlp = fuse("mlp", spec(MLP_FUSION))
    weighted_avg = fuse("weighted_avg", spec(WAVG_FUSION))

    return [
        extract(
            name="extract_fusion",
            dataset=dataset,
            extractor_ckpts={"vgae": vgae_ckpt, "gat": gat_ckpt},
            output_dir=extract_dir,
            seed=seed,
        ),
        bandit.fit("bandit"),             bandit.test("bandit"),
        dqn.fit("dqn"),                   dqn.test("dqn"),
        mlp.fit("mlp"),                   mlp.test("mlp"),
        weighted_avg.fit("weighted_avg"), weighted_avg.test("weighted_avg"),
    ]
