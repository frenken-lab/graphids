"""Per-checkpoint artifact generation for all trained ablation checkpoints.

Emits one AnalyzeRow per checkpoint available for the given dataset + seed:
  - VGAE (unsupervised/vgae, teacher/teacher_vgae): embeddings + landscape
  - GAT  (teacher_gat + all ablation GATs):          embeddings + attention + landscape
           + CKA vs teacher_gat for all ablation GATs (not teacher itself)
  - Fusion (all fusion variants):                    fusion_policy only

Prerequisites:
  - All fit rows for the dataset must be FINISHED (check MLflow before rendering)
  - GRAPHIDS_LAKE_ROOT and GRAPHIDS_RUN_ROOT must be set at render time

Usage:
    gx run ops.analyze -d hcrl_sa -s 42 -o rendered/hcrl_sa/ops/analyze/seed_42.json
    gx plans submit --plan rendered/hcrl_sa/ops/analyze/seed_42.json -C pitzer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphids.paths import best_ckpt, lake_root, run_dir
from graphids.plan.compose import analyze

_LAKE_ROOT = lake_root()

_VGAE_VARIANTS: list[tuple[str, str]] = [
    ("unsupervised", "vgae"),
    ("teacher", "teacher_vgae"),
]

_GAT_VARIANTS: list[tuple[str, str]] = [
    ("teacher", "teacher_gat"),
    ("gat_loss", "ce"),
    ("gat_loss", "focal"),
    ("gat_loss", "weighted_ce"),
    ("gat_sampling", "curriculum_random"),
    ("gat_sampling", "curriculum_vgae"),
    ("gat_sampling", "none"),
    ("id_encoding", "hash"),
    ("id_encoding", "lookup"),
]

_FUSION_VARIANTS: list[str] = [
    "bandit",
    "dqn",
    "mlp",
    "moe",
    "moe_noaux",
    "weighted_avg",
]


def _exists(ckpt: str) -> bool:
    return Path(ckpt).exists()


def build(*, dataset: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    teacher_gat_ckpt = best_ckpt(dataset, "teacher", "teacher_gat", seed)

    # ── VGAE: embeddings + landscape — CPU long (21×21 = 441 passes) ────────
    for group, variant in _VGAE_VARIANTS:
        ckpt = best_ckpt(dataset, group, variant, seed)
        if not _exists(ckpt):
            continue
        out = str(Path(run_dir(dataset, group, variant, seed)) / "artifacts")
        rows.append(
            analyze(
                name=f"analyze_{variant}_{dataset}",
                ckpt_path=ckpt,
                dataset=dataset,
                model_type="vgae",
                output_dir=out,
                lake_root=_LAKE_ROOT,
                embeddings=True,
                landscape=True,
                landscape_resolution=21,
                seed=seed,
                mode="cpu",
                length="long",
            )
        )

    # ── GAT: embeddings + attention + landscape (+ CKA) — CPU long ──────────
    if not _exists(teacher_gat_ckpt):
        import warnings

        warnings.warn(
            f"teacher_gat ckpt missing for {dataset}/seed_{seed} — skipping all GAT rows",
            stacklevel=2,
        )
    else:
        for group, variant in _GAT_VARIANTS:
            ckpt = best_ckpt(dataset, group, variant, seed)
            if not _exists(ckpt):
                continue
            out = str(Path(run_dir(dataset, group, variant, seed)) / "artifacts")
            is_teacher = variant == "teacher_gat"
            rows.append(
                analyze(
                    name=f"analyze_{variant}_{dataset}",
                    ckpt_path=ckpt,
                    dataset=dataset,
                    model_type="gat",
                    output_dir=out,
                    lake_root=_LAKE_ROOT,
                    embeddings=True,
                    attention=True,
                    landscape=True,
                    landscape_resolution=21,
                    cka=not is_teacher,
                    cka_teacher_ckpt="" if is_teacher else teacher_gat_ckpt,
                    seed=seed,
                    mode="cpu",
                    length="long",
                )
            )

    # ── Fusion: fusion_policy only — CPU short (reads pre-extracted states) ──
    vgae_ckpt = best_ckpt(dataset, "unsupervised", "vgae", seed)
    gat_ckpt = best_ckpt(dataset, "gat_loss", "focal", seed)
    for variant in _FUSION_VARIANTS:
        ckpt = best_ckpt(dataset, "fusion", variant, seed)
        if not _exists(ckpt):
            continue
        out = str(Path(run_dir(dataset, "fusion", variant, seed)) / "artifacts")
        rows.append(
            analyze(
                name=f"analyze_{variant}_{dataset}",
                ckpt_path=ckpt,
                dataset=dataset,
                model_type="fusion",
                output_dir=out,
                lake_root=_LAKE_ROOT,
                embeddings=False,
                fusion_policy=True,
                vgae_ckpt_path=vgae_ckpt,
                gat_ckpt_path=gat_ckpt,
                seed=seed,
                mode="cpu",
                length="short",
            )
        )

    return rows
