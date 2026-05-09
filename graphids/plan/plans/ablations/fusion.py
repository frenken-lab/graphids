"""Fusion plan.

Emits one extract row + 5 method-specific (fit, test) pairs. Submit:
extract first, then chain method rows via ``--depends-on-afterok``.
"""

from __future__ import annotations

from typing import Any

from graphids.paths import best_ckpt, states_dir
from graphids.plan import (
    FUSION_TRAINER,
    REWARD,
    bandit,
    dqn,
    extract,
    fit_row,
    fusion_dm,
    mlp_fusion,
    moe,
    test_row,
    weighted_avg,
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

    def fuse(variant: str, model) -> dict[str, Any]:
        return dict(
            model=model,
            data=fusion_dm(dataset=dataset, seed=seed, method=variant),
            meta=meta(variant),
            monitor="val_acc",
            mode="max",
            run_mode="cpu",
            trainer_overrides=FUSION_TRAINER,
            upstreams=[
                {"role": "vgae", "ckpt_path": vgae_ckpt, "ckpt_tla": "vgae_ckpt_path"},
                {"role": "gat", "ckpt_path": gat_ckpt, "ckpt_tla": "gat_ckpt_path"},
            ],
            patience=40,
        )

    # state_dim = 18: VGAE (errors[3]+conf[1]+z_stats[4]+spike[1]+affinity[1]+rq[1]=11)
    #                + GAT (probs[2]+conf[1]+emb_stats[4]=7) — sorted leaf concat in flatten_features.
    _state_dim = 18
    bandit_kw = fuse("bandit", bandit(state_dim=_state_dim, reward_kwargs=dict(REWARD)))
    dqn_kw = fuse("dqn", dqn(state_dim=_state_dim, reward_kwargs=dict(REWARD)))
    mlp_kw = fuse("mlp", mlp_fusion(state_dim=_state_dim))
    moe_kw = fuse("moe", moe(state_dim=_state_dim))
    # Aux-loss ablation row — same model, aux_weight=0.0 (no load-balance pressure).
    # Direct test of whether the Switch-style L_aux is the load-bearing piece for MoE.
    moe_noaux_kw = fuse("moe_noaux", moe(state_dim=_state_dim, aux_weight=0.0))
    weighted_avg_kw = fuse("weighted_avg", weighted_avg(state_dim=_state_dim))

    return [
        extract(
            name="extract_fusion",
            dataset=dataset,
            extractor_ckpts={"vgae": vgae_ckpt, "gat": gat_ckpt},
            output_dir=extract_dir,
            seed=seed,
        ),
        fit_row("bandit", **bandit_kw),
        test_row("bandit", **bandit_kw),
        fit_row("dqn", **dqn_kw),
        test_row("dqn", **dqn_kw),
        fit_row("mlp", **mlp_kw),
        test_row("mlp", **mlp_kw),
        fit_row("moe", **moe_kw),
        test_row("moe", **moe_kw),
        fit_row("moe_noaux", **moe_noaux_kw),
        test_row("moe_noaux", **moe_noaux_kw),
        fit_row("weighted_avg", **weighted_avg_kw),
        test_row("weighted_avg", **weighted_avg_kw),
    ]
