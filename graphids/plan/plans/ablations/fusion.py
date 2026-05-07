"""Fusion plan.

Emits one extract row + 5 method-specific (fit, test) pairs. Submit:
extract first, then chain method rows via ``--depends-on-afterok``.
"""

from __future__ import annotations

from typing import Any

from graphids.paths import best_ckpt, states_dir
from graphids.plan import (
    BANDIT,
    DQN,
    MLP_FUSION,
    MOE_FUSION,
    REWARD,
    WAVG_FUSION,
    extract,
    fusion,
    spec,
)


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

    # state_dim = 18: VGAE (errors[3]+conf[1]+z_stats[4]+spike[1]+affinity[1]+rq[1]=11)
    #                + GAT (probs[2]+conf[1]+emb_stats[4]=7) — sorted leaf concat in flatten_features.
    _state_dim = 18
    bandit = fuse("bandit", spec(BANDIT, state_dim=_state_dim, reward_kwargs=dict(REWARD)))
    dqn = fuse("dqn", spec(DQN, state_dim=_state_dim, reward_kwargs=dict(REWARD)))
    mlp = fuse("mlp", spec(MLP_FUSION, state_dim=_state_dim))
    moe = fuse("moe", spec(MOE_FUSION, state_dim=_state_dim))
    # Aux-loss ablation row — same model, aux_weight=0.0 (no load-balance pressure).
    # Direct test of whether the Switch-style L_aux is the load-bearing piece for MoE.
    moe_noaux = fuse("moe_noaux", spec(MOE_FUSION, state_dim=_state_dim, aux_weight=0.0))
    weighted_avg = fuse("weighted_avg", spec(WAVG_FUSION, state_dim=_state_dim))

    return [
        extract(
            name="extract_fusion",
            dataset=dataset,
            extractor_ckpts={"vgae": vgae_ckpt, "gat": gat_ckpt},
            output_dir=extract_dir,
            seed=seed,
        ),
        bandit.fit("bandit"),
        bandit.test("bandit"),
        dqn.fit("dqn"),
        dqn.test("dqn"),
        mlp.fit("mlp"),
        mlp.test("mlp"),
        moe.fit("moe"),
        moe.test("moe"),
        moe_noaux.fit("moe_noaux"),
        moe_noaux.test("moe_noaux"),
        weighted_avg.fit("weighted_avg"),
        weighted_avg.test("weighted_avg"),
    ]
