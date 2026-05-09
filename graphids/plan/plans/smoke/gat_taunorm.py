"""GAT + TauNorm Lightning-migration smoke.

max_epochs=2 keeps walltime <= 10 min on Pitzer V100.
"""

from __future__ import annotations

from typing import Any

from graphids.plan import can_bus, fit_row, focal, gat, graph_dm, test_row


def build(*, dataset: str, seed: int) -> list[dict[str, Any]]:
    kw = dict(
        model=gat(),
        data=graph_dm(source=can_bus(dataset=dataset, seed=seed)),
        loss=focal(),
        meta={
            "group": "lightning_migration_smoke",
            "variant": "gat_taunorm",
            "dataset": dataset,
            "seed": seed,
            "model_type": "gat",
            "scale": "small",
        },
        trainer_overrides={"max_epochs": 2},
        callback_extras={
            "kang_tau_norm": {
                "class_path": "graphids.core.callbacks.TauNormCallback",
                "init_args": {"tau": 0.5},
            },
        },
    )
    return [fit_row("gat_taunorm", **kw), test_row("gat_taunorm", **kw)]
