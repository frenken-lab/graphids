"""Main training run — full 3-stage KD chain.

Phase 1  Teacher (large VGAE + large GAT, no KD)
Phase 2  Student with KD (small, distilled from Phase 1 ckpts)
Phase 3  Student without KD (small, no distillation — A/B baseline)
Phase 4  Fusion on both student variants (two independent extract+fuse tracks)

Dependency chain for SLURM submission:

  teacher_vgae ──► student_vgae_kd ──► extract_fusion_kd ──► fusion_*_kd
  teacher_gat  ──► student_gat_kd  ──┘
  student_vgae_nokd ──► extract_fusion_nokd ──► fusion_*_nokd
  student_gat_nokd  ──┘

The no-KD student rows have no upstream dependency and can run in parallel
with teacher training from submission time.
"""

from __future__ import annotations

from typing import Any

from graphids.paths import best_ckpt, states_dir
from graphids.plan import (
    FUSION_TRAINER,
    REWARD,
    bandit,
    can_bus,
    dqn,
    extract,
    feature_distillation,
    fit_row,
    focal,
    fusion_dm,
    gat,
    graph_dm,
    mlp_fusion,
    moe,
    soft_label_distillation,
    test_row,
    vgae,
    vgae_task,
    weighted_avg,
)

# VGAE (11 features) + GAT (7 features) — see ablations/fusion.py for breakdown.
_STATE_DIM = 18

_FUSION_METHODS = [
    ("mlp", mlp_fusion(state_dim=_STATE_DIM)),
    ("moe", moe(state_dim=_STATE_DIM)),
    ("moe_noaux", moe(state_dim=_STATE_DIM, aux_weight=0.0)),
    ("bandit", bandit(state_dim=_STATE_DIM, reward_kwargs=dict(REWARD))),
    ("dqn", dqn(state_dim=_STATE_DIM, reward_kwargs=dict(REWARD))),
    ("weighted_avg", weighted_avg(state_dim=_STATE_DIM)),
]


def build(*, dataset: str, seed: int) -> list[dict[str, Any]]:
    # ------------------------------------------------------------------ ckpt paths
    teacher_vgae_ckpt = best_ckpt(dataset, "teacher", "teacher_vgae", seed, subdir="training")
    teacher_gat_ckpt = best_ckpt(dataset, "teacher", "teacher_gat", seed, subdir="training")
    student_vgae_kd_ckpt = best_ckpt(
        dataset, "student_kd", "student_vgae_kd", seed, subdir="training"
    )
    student_gat_kd_ckpt = best_ckpt(
        dataset, "student_kd", "student_gat_kd", seed, subdir="training"
    )
    student_vgae_nokd_ckpt = best_ckpt(
        dataset, "student_nokd", "student_vgae_nokd", seed, subdir="training"
    )
    student_gat_nokd_ckpt = best_ckpt(
        dataset, "student_nokd", "student_gat_nokd", seed, subdir="training"
    )

    # ------------------------------------------------------------------ data helpers
    def vgae_data():
        return graph_dm(
            source=can_bus(dataset=dataset, seed=seed),
            label_filter="benign",
            min_steps_per_epoch=50,
        )

    def gat_data():
        return graph_dm(source=can_bus(dataset=dataset, seed=seed))

    # ------------------------------------------------------------------ meta helpers
    def meta(group: str, variant: str, model_type: str, scale: str) -> dict[str, Any]:
        return {
            "group": group,
            "variant": variant,
            "dataset": dataset,
            "seed": seed,
            "model_type": model_type,
            "scale": scale,
            "subdir": "training",
        }

    # ------------------------------------------------------------------ fusion helper
    def fuse_track(method: str, model, *, track: str, vgae_ckpt: str, gat_ckpt: str):
        """Build kw dict for one fusion row with explicit upstream ckpts and states_variant."""
        variant = f"{method}_{track}"
        return dict(
            model=model,
            data=fusion_dm(dataset=dataset, seed=seed, method=method, states_variant=track),
            meta=meta(f"fusion_{track}", variant, "fusion", "small"),
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

    rows: list[dict[str, Any]] = []

    # ====================================================== Phase 1 — Teacher
    teacher_vgae_kw = dict(
        model=vgae(scale="large"),
        data=vgae_data(),
        loss=vgae_task(),
        monitor="val_discrimination_ratio",
        meta=meta("teacher", "teacher_vgae", "vgae", "large"),
        patience=200,
        trainer_overrides={"max_epochs": 600, "precision": "32-true"},
    )
    rows += [
        fit_row("teacher_vgae", **teacher_vgae_kw),
        test_row("teacher_vgae", **teacher_vgae_kw),
    ]

    teacher_gat_kw = dict(
        model=gat(scale="large"),
        data=gat_data(),
        loss=focal(),
        meta=meta("teacher", "teacher_gat", "gat", "large"),
        trainer_overrides={"max_epochs": 200},
    )
    rows += [fit_row("teacher_gat", **teacher_gat_kw), test_row("teacher_gat", **teacher_gat_kw)]

    # ====================================================== Phase 2 — Student with KD
    student_vgae_kd_kw = dict(
        model=vgae(scale="small"),
        data=vgae_data(),
        loss=feature_distillation(
            base_loss=vgae_task(),
            teacher_model=vgae(scale="large"),
            teacher_ckpt_path=teacher_vgae_ckpt,
            projection_in_features=64,
            projection_out_features=128,
        ),
        monitor="val_discrimination_ratio",
        meta=meta("student_kd", "student_vgae_kd", "vgae", "small"),
        patience=200,
        trainer_overrides={"max_epochs": 600, "precision": "32-true"},
        upstreams=[
            {
                "role": "teacher_vgae",
                "ckpt_path": teacher_vgae_ckpt,
                "ckpt_tla": "teacher_vgae_ckpt",
            }
        ],
    )
    rows += [
        fit_row("student_vgae_kd", **student_vgae_kd_kw),
        test_row("student_vgae_kd", **student_vgae_kd_kw),
    ]

    student_gat_kd_kw = dict(
        model=gat(scale="small"),
        data=gat_data(),
        loss=soft_label_distillation(
            base_loss=focal(),
            teacher_model=gat(scale="large"),
            teacher_ckpt_path=teacher_gat_ckpt,
        ),
        meta=meta("student_kd", "student_gat_kd", "gat", "small"),
        trainer_overrides={"max_epochs": 200},
        upstreams=[
            {
                "role": "teacher_gat",
                "ckpt_path": teacher_gat_ckpt,
                "ckpt_tla": "teacher_gat_ckpt",
            }
        ],
    )
    rows += [
        fit_row("student_gat_kd", **student_gat_kd_kw),
        test_row("student_gat_kd", **student_gat_kd_kw),
    ]

    # ====================================================== Phase 3 — Student without KD
    student_vgae_nokd_kw = dict(
        model=vgae(scale="small"),
        data=vgae_data(),
        loss=vgae_task(),
        monitor="val_discrimination_ratio",
        meta=meta("student_nokd", "student_vgae_nokd", "vgae", "small"),
        patience=200,
        trainer_overrides={"max_epochs": 600, "precision": "32-true"},
    )
    rows += [
        fit_row("student_vgae_nokd", **student_vgae_nokd_kw),
        test_row("student_vgae_nokd", **student_vgae_nokd_kw),
    ]

    student_gat_nokd_kw = dict(
        model=gat(scale="small"),
        data=gat_data(),
        loss=focal(),
        meta=meta("student_nokd", "student_gat_nokd", "gat", "small"),
        trainer_overrides={"max_epochs": 200},
    )
    rows += [
        fit_row("student_gat_nokd", **student_gat_nokd_kw),
        test_row("student_gat_nokd", **student_gat_nokd_kw),
    ]

    # ====================================================== Phase 4 — Fusion (two tracks)
    for track, vgae_ckpt, gat_ckpt in [
        ("kd", student_vgae_kd_ckpt, student_gat_kd_ckpt),
        ("nokd", student_vgae_nokd_ckpt, student_gat_nokd_ckpt),
    ]:
        rows.append(
            extract(
                name=f"extract_fusion_{track}",
                dataset=dataset,
                extractor_ckpts={"vgae": vgae_ckpt, "gat": gat_ckpt},
                output_dir=states_dir(dataset, seed, track),
                seed=seed,
            )
        )
        for method, model in _FUSION_METHODS:
            kw = fuse_track(method, model, track=track, vgae_ckpt=vgae_ckpt, gat_ckpt=gat_ckpt)
            variant = f"{method}_{track}"
            rows += [fit_row(variant, **kw), test_row(variant, **kw)]

    return rows
