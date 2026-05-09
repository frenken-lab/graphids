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
    BANDIT,
    DQN,
    FEATURE_DISTILLATION,
    FOCAL,
    GAT,
    MLP_FUSION,
    MOE_FUSION,
    REWARD,
    SOFT_LABEL_DISTILLATION,
    VGAE,
    VGAE_TASK,
    WAVG_FUSION,
    can_bus,
    compose,
    extract,
    fusion_dm,
    graph_dm,
    spec,
)

# Matches _FUSION_TRAINER_OVERLAY in compose.py — reproduced here because
# we call compose() directly to supply custom upstream ckpt paths.
_FUSION_TRAINER: dict[str, Any] = {
    "precision": "32-true",
    "gradient_clip_val": None,
    "max_epochs": 1500,
    "log_every_n_steps": 50,
    "check_val_every_n_epoch": 5,
    "reload_dataloaders_every_n_epochs": 1,
}

# VGAE (11 features) + GAT (7 features) — see ablations/fusion.py for breakdown.
_STATE_DIM = 18

_FUSION_METHODS: list[tuple[str, dict[str, Any]]] = [
    ("mlp", spec(MLP_FUSION, state_dim=_STATE_DIM)),
    ("moe", spec(MOE_FUSION, state_dim=_STATE_DIM)),
    ("moe_noaux", spec(MOE_FUSION, state_dim=_STATE_DIM, aux_weight=0.0)),
    ("bandit", spec(BANDIT, state_dim=_STATE_DIM, reward_kwargs=dict(REWARD))),
    ("dqn", spec(DQN, state_dim=_STATE_DIM, reward_kwargs=dict(REWARD))),
    ("weighted_avg", spec(WAVG_FUSION, state_dim=_STATE_DIM)),
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
    def vgae_data() -> dict[str, Any]:
        return graph_dm(
            source=can_bus(dataset=dataset, seed=seed),
            label_filter="benign",
            min_steps_per_epoch=50,
        )

    def gat_data() -> dict[str, Any]:
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
    def fuse_track(
        method: str,
        model: dict[str, Any],
        *,
        track: str,
        vgae_ckpt: str,
        gat_ckpt: str,
    ) -> Any:
        """Compose one fusion row with explicit upstream ckpts and states_variant."""
        variant = f"{method}_{track}"
        return compose(
            model=model,
            data=fusion_dm(dataset=dataset, seed=seed, method=method, states_variant=track),
            meta=meta(f"fusion_{track}", variant, "fusion", "small"),
            monitor="val_acc",
            mode="max",
            run_mode="cpu",
            trainer_overrides=_FUSION_TRAINER,
            upstreams=[
                {"role": "vgae", "ckpt_path": vgae_ckpt, "ckpt_tla": "vgae_ckpt_path"},
                {"role": "gat", "ckpt_path": gat_ckpt, "ckpt_tla": "gat_ckpt_path"},
            ],
            patience=40,
        )

    rows: list[dict[str, Any]] = []

    # ====================================================== Phase 1 — Teacher
    teacher_vgae = compose(
        model=spec(VGAE, scale="large"),
        data=vgae_data(),
        loss=spec(VGAE_TASK),
        monitor="val_discrimination_ratio",
        meta=meta("teacher", "teacher_vgae", "vgae", "large"),
        patience=200,
        trainer_overrides={"max_epochs": 600, "precision": "32-true"},
    )
    rows += [teacher_vgae.fit("teacher_vgae"), teacher_vgae.test("teacher_vgae")]

    teacher_gat = compose(
        model=spec(GAT, scale="large"),
        data=gat_data(),
        loss=spec(FOCAL),
        meta=meta("teacher", "teacher_gat", "gat", "large"),
        trainer_overrides={"max_epochs": 200},
    )
    rows += [teacher_gat.fit("teacher_gat"), teacher_gat.test("teacher_gat")]

    # ====================================================== Phase 2 — Student with KD
    student_vgae_kd = compose(
        model=spec(VGAE, scale="small"),
        data=vgae_data(),
        loss=spec(
            FEATURE_DISTILLATION,
            base_loss=spec(VGAE_TASK),
            teacher_model=spec(VGAE, scale="large"),
            teacher_ckpt_path=teacher_vgae_ckpt,
            projection=spec("torch.nn.Linear", in_features=64, out_features=128),
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
    rows += [student_vgae_kd.fit("student_vgae_kd"), student_vgae_kd.test("student_vgae_kd")]

    student_gat_kd = compose(
        model=spec(GAT, scale="small"),
        data=gat_data(),
        loss=spec(
            SOFT_LABEL_DISTILLATION,
            base_loss=spec(FOCAL),
            teacher_model=spec(GAT, scale="large"),
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
    rows += [student_gat_kd.fit("student_gat_kd"), student_gat_kd.test("student_gat_kd")]

    # ====================================================== Phase 3 — Student without KD
    student_vgae_nokd = compose(
        model=spec(VGAE, scale="small"),
        data=vgae_data(),
        loss=spec(VGAE_TASK),
        monitor="val_discrimination_ratio",
        meta=meta("student_nokd", "student_vgae_nokd", "vgae", "small"),
        patience=200,
        trainer_overrides={"max_epochs": 600, "precision": "32-true"},
    )
    rows += [
        student_vgae_nokd.fit("student_vgae_nokd"),
        student_vgae_nokd.test("student_vgae_nokd"),
    ]

    student_gat_nokd = compose(
        model=spec(GAT, scale="small"),
        data=gat_data(),
        loss=spec(FOCAL),
        meta=meta("student_nokd", "student_gat_nokd", "gat", "small"),
        trainer_overrides={"max_epochs": 200},
    )
    rows += [student_gat_nokd.fit("student_gat_nokd"), student_gat_nokd.test("student_gat_nokd")]

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
            row = fuse_track(method, model, track=track, vgae_ckpt=vgae_ckpt, gat_ckpt=gat_ckpt)
            variant = f"{method}_{track}"
            rows += [row.fit(variant), row.test(variant)]

    return rows
