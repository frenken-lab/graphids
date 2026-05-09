"""Unsupervised plan."""

from __future__ import annotations

from typing import Any

from graphids.plan import can_bus, dgi, fit_row, graph_dm, test_row, vgae, vgae_task


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
    vgae_data = graph_dm(
        source=can_bus(dataset=dataset, seed=seed),
        label_filter="benign",
        min_steps_per_epoch=50,
    )

    vgae_kw = dict(
        model=vgae(),
        data=vgae_data,
        loss=vgae_task(),
        monitor="val_recon_max_gap",
        meta=meta("vgae", "vgae"),
        patience=200,
        trainer_overrides={"max_epochs": 600, "precision": "32-true"},
    )
    dgi_kw = dict(
        model=dgi(),
        data=data,
        monitor="val_dgi_loss",
        meta=meta("dgi", "dgi"),
        trainer_overrides={"max_epochs": 400},
    )
    return [
        fit_row("vgae", **vgae_kw),
        test_row("vgae", **vgae_kw),
        fit_row("dgi", **dgi_kw),
        test_row("dgi", **dgi_kw),
    ]
