"""Unsupervised plan."""

from __future__ import annotations

from typing import Any

from graphids.graphids.config.compose import compose
from graphids.graphids.config.lib import DGI, VGAE, VGAE_TASK, can_bus, graph_dm, spec


def build(*, dataset: str, seed: int) -> list[dict[str, Any]]:
    def meta(variant: str, mt: str) -> dict[str, Any]:
        return {
            "group": "unsupervised",
            "variant": variant,
            "dataset": dataset,
            "seed": seed,
            "model_type": mt,
            "scale": "small",
        }

    data = graph_dm(source=can_bus(dataset=dataset, seed=seed), label_filter="benign")

    vgae = compose(
        model=spec(VGAE),
        data=data,
        loss=spec(VGAE_TASK),
        monitor="val_discrimination_ratio",
        meta=meta("vgae", "vgae"),
        trainer_overrides={"max_epochs": 600, "precision": "32-true"},
    )
    dgi = compose(
        model=spec(DGI),
        data=data,
        monitor="val_dgi_loss",
        meta=meta("dgi", "dgi"),
        trainer_overrides={"max_epochs": 400},
    )
    return [
        vgae.fit("vgae"), vgae.test("vgae"),
        dgi.fit("dgi"),   dgi.test("dgi"),
    ]
