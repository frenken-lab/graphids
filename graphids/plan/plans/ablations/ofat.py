"""OFAT (one-factor-at-a-time) plan — declarative axis sweeps.

The full ablation matrix lives in :data:`SWEEPS` (built inside
``build()`` so variants can close over ``dataset`` / ``seed`` / the
upstream VGAE checkpoint). Adding an ablation row = adding a dict
entry; the row builder + emit loop don't change.

Variant-key → fit/test row name is the variant key itself unless the
entry carries an explicit ``"name"`` (used today for ``scaler/*`` and
``id_encoding/*`` to avoid collisions with other axes).

Subsumes the prior ``supervised`` / ``supervised_ablations`` plans —
those were 1-row and 3-fit-only slices of this matrix. Reintroduce
slicing later via a ``--variants`` CLI flag if needed.
"""

from __future__ import annotations

from typing import Any

from graphids.paths import best_ckpt
from graphids.plan import (
    CE,
    FOCAL,
    GAT,
    SCORE_RANDOM,
    SCORE_VGAE,
    VGAE,
    VGAE_TASK,
    WEIGHTED_CE,
    can_bus,
    compose,
    curriculum,
    graph_dm,
    spec,
)


def build(*, dataset: str, seed: int) -> list[dict[str, Any]]:
    vgae_ckpt = best_ckpt(dataset, "unsupervised", "vgae", seed)

    def gat_meta(group: str, variant: str) -> dict[str, Any]:
        return {
            "group": group, "variant": variant,
            "dataset": dataset, "seed": seed,
            "model_type": "gat", "scale": "small",
        }

    def gat_row(
        group: str,
        variant: str,
        *,
        loss: dict[str, Any],
        difficulty: dict[str, Any] | None = None,
        source_overrides: dict[str, Any] | None = None,
        model_init_extra: dict[str, Any] | None = None,
        upstreams: list[dict[str, Any]] | None = None,
        trainer_overrides: dict[str, Any] | None = None,
    ):
        return compose(
            model=spec(GAT, **(model_init_extra or {})),
            data=graph_dm(
                source=can_bus(dataset=dataset, seed=seed, **(source_overrides or {})),
                difficulty=difficulty,
            ),
            loss=loss,
            meta=gat_meta(group, variant),
            upstreams=upstreams,
            trainer_overrides=trainer_overrides or {},
        )

    SWEEPS: dict[str, dict[str, dict[str, Any]]] = {
        "gat_loss": {
            "focal":       {"loss": spec(FOCAL),
                            "trainer_overrides": {"max_epochs": 200}},
            "ce":          {"loss": spec(CE)},
            "weighted_ce": {"loss": spec(WEIGHTED_CE, weights=[1.0, 5.0])},
        },
        "gat_sampling": {
            "none":              {"loss": spec(FOCAL)},
            "curriculum_random": {"loss": curriculum(spec(FOCAL)),
                                  "difficulty": spec(SCORE_RANDOM, seed=seed)},
            "curriculum_vgae":   {"loss": curriculum(spec(FOCAL)),
                                  "difficulty": spec(SCORE_VGAE, ckpt_path=vgae_ckpt),
                                  "upstreams": [{"role": "vgae",
                                                 "ckpt_path": vgae_ckpt,
                                                 "ckpt_tla": "vgae_ckpt_path"}]},
        },
        "scaler": {
            "z_benign":      {"name": "scaler_z",      "loss": spec(FOCAL),
                              "source_overrides": {"scaler_strategy": "z_benign"}},
            "robust_benign": {"name": "scaler_robust", "loss": spec(FOCAL),
                              "source_overrides": {"scaler_strategy": "robust_benign"}},
        },
        "id_encoding": {
            "lookup": {"name": "id_lookup", "loss": spec(FOCAL),
                       "model_init_extra": {
                           "id_encoder_class_path":
                               "graphids.core.models.id_encoding.lookup.LookupIdEncoder",
                       }},
            "hash":   {"name": "id_hash",   "loss": spec(FOCAL),
                       "model_init_extra": {
                           "id_encoder_class_path":
                               "graphids.core.models.id_encoding.hash_embedding.HashIdEncoder",
                           "id_encoder_kwargs": {"num_buckets": 2048},
                       }},
        },
    }

    # ----- unsupervised VGAE baseline (produces the ckpt curriculum_vgae reads) -
    vgae = compose(
        model=spec(VGAE),
        data=graph_dm(source=can_bus(dataset=dataset, seed=seed), label_filter="benign"),
        loss=spec(VGAE_TASK),
        monitor="val_discrimination_ratio",
        meta={
            "group": "unsupervised", "variant": "vgae",
            "dataset": dataset, "seed": seed,
            "model_type": "vgae", "scale": "small",
        },
        trainer_overrides={"max_epochs": 600, "precision": "32-true"},
    )
    rows: list[dict[str, Any]] = [vgae.fit("vgae"), vgae.test("vgae")]

    # ----- emit the sweep (fit + test per variant) ----------------------------
    for group, variants in SWEEPS.items():
        for variant, kwargs in variants.items():
            name = kwargs.pop("name", variant)
            row = gat_row(group, variant, **kwargs)
            rows.extend([row.fit(name), row.test(name)])
    return rows
