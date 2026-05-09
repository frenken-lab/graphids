"""Supervised OFAT (one-factor-at-a-time) plan — declarative axis sweeps.

The full ablation matrix lives in :data:`SWEEPS` (built inside
``build()`` so variants can close over ``dataset`` / ``seed`` / the
upstream VGAE checkpoint). Adding an ablation row = adding a dict
entry; the row builder + emit loop don't change.

Variant-key → fit/test row name is the variant key itself unless the
entry carries an explicit ``"name"`` (used today for ``id_encoding/*``
to avoid collisions with other axes).
"""

from __future__ import annotations

from typing import Any

from graphids.paths import best_ckpt
from graphids.plan import (
    can_bus,
    ce,
    curriculum,
    fit_row,
    focal,
    gat,
    graph_dm,
    score_random,
    score_vgae,
    test_row,
    vgae,
    vgae_task,
    weighted_ce,
)


def build(*, dataset: str, seed: int) -> list[dict[str, Any]]:
    vgae_ckpt = best_ckpt(dataset, "unsupervised", "vgae", seed)

    def gat_meta(group: str, variant: str) -> dict[str, Any]:
        return {
            "group": group,
            "variant": variant,
            "dataset": dataset,
            "seed": seed,
            "model_type": "gat",
            "scale": "small",
        }

    def gat_row(
        group: str,
        variant: str,
        *,
        loss,
        difficulty=None,
        source_overrides: dict[str, Any] | None = None,
        model_init_extra: dict[str, Any] | None = None,
        upstreams: list[dict[str, Any]] | None = None,
        trainer_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return dict(
            model=gat(**(model_init_extra or {})),
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
            "focal": {"loss": focal(), "trainer_overrides": {"max_epochs": 200}},
            "ce": {"loss": ce()},
            "weighted_ce": {"loss": weighted_ce(weights=[1.0, 5.0])},
        },
        "gat_sampling": {
            "none": {"loss": focal()},
            "curriculum_random": {
                "loss": curriculum(focal()),
                "difficulty": score_random(seed=seed),
            },
            "curriculum_vgae": {
                "loss": curriculum(focal()),
                "difficulty": score_vgae(vgae_ckpt),
                "upstreams": [
                    {"role": "vgae", "ckpt_path": vgae_ckpt, "ckpt_tla": "vgae_ckpt_path"}
                ],
            },
        },
        "id_encoding": {
            "lookup": {
                "name": "id_lookup",
                "loss": focal(),
                "model_init_extra": {
                    "id_encoder_class_path": "graphids.core.models.id_encoding.lookup.LookupIdEncoder",
                },
            },
            "hash": {
                "name": "id_hash",
                "loss": focal(),
                "model_init_extra": {
                    "id_encoder_class_path": "graphids.core.models.id_encoding.hash_embedding.HashIdEncoder",
                    "id_encoder_kwargs": {"num_buckets": 2048},
                },
            },
        },
    }

    # ----- unsupervised VGAE baseline (produces the ckpt curriculum_vgae reads) -
    vgae_kw = dict(
        model=vgae(),
        data=graph_dm(source=can_bus(dataset=dataset, seed=seed), label_filter="benign"),
        loss=vgae_task(),
        monitor="val_discrimination_ratio",
        meta={
            "group": "unsupervised",
            "variant": "vgae",
            "dataset": dataset,
            "seed": seed,
            "model_type": "vgae",
            "scale": "small",
        },
        trainer_overrides={"max_epochs": 600, "precision": "32-true"},
    )
    rows: list[dict[str, Any]] = [fit_row("vgae", **vgae_kw), test_row("vgae", **vgae_kw)]

    # ----- emit the sweep (fit + test per variant) ----------------------------
    for group, variants in SWEEPS.items():
        for variant, kwargs in variants.items():
            name = kwargs.pop("name", variant)
            kw = gat_row(group, variant, **kwargs)
            rows.extend([fit_row(name, **kw), test_row(name, **kw)])
    return rows
