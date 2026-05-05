"""GAT + TauNorm Lightning-migration smoke.

max_epochs=2 keeps walltime <= 10 min on Pitzer V100.
"""

from __future__ import annotations

from typing import Any

from graphids.graphids.config.compose import compose
from graphids.graphids.config.lib import FOCAL, GAT, can_bus, graph_dm, spec


def build(*, dataset: str, seed: int) -> list[dict[str, Any]]:
    gat_smoke = compose(
        model=spec(GAT),
        data=graph_dm(source=can_bus(dataset=dataset, seed=seed)),
        loss=spec(FOCAL),
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
    return [gat_smoke.fit("gat_taunorm"), gat_smoke.test("gat_taunorm")]
