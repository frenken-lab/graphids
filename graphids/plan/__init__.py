"""GraphIDS plan layer — Python plans → typed rows → JSON.

Pipeline (sort the files alphabetically; they read top-down)::

    plans/<name>.py::build()              ← you write this
        └── primitives  (spec, GAT, …)    ← leaves          (primitives.py)
              └── compose() / fusion()    ← apex builder    (compose.py)
                    └── RowSpec.fit/test  ← row emitter     (compose.py)
                          └── schema.Plan ← typed contract  (schema.py)
                                └── JSON → graphids exec

Plan authors should import from this package — internal modules are an
implementation detail.

    from graphids.plan import (
        compose, fusion, extract,           # apex + one-shot row builders
        spec, can_bus, graph_dm, curriculum,  # primitives
        GAT, VGAE, FOCAL, CE, …,            # class-path constants
    )

The ``schema`` module (``Plan``, ``TrainRow``, …) is the validation
contract for ``graphids run``/``exec``/``submit``; plan authors do not
import from it directly.
"""

from graphids.plan.compose import (
    RowSpec,
    analyze,
    callbacks_base,
    compose,
    extract,
    fusion,
    trainer_base,
)
from graphids.plan.primitives import (
    BANDIT,
    CAN_BUS,
    CE,
    CURRICULUM_LOSS,
    DGI,
    DQN,
    FEATURE_DISTILLATION,
    FOCAL,
    FUSION_DM,
    GAT,
    GRAPH_DM,
    LINEAR_RAMP,
    MLP_FUSION,
    MOE_FUSION,
    REWARD,
    REWARD_MINIMAL,
    SCORE_RANDOM,
    SCORE_VGAE,
    SOFT_LABEL_DISTILLATION,
    VGAE,
    VGAE_TASK,
    WAVG_FUSION,
    WEIGHTED_CE,
    can_bus,
    curriculum,
    fusion_dm,
    graph_dm,
    spec,
)

__all__ = [
    # compose
    "RowSpec",
    "analyze",
    "compose",
    "fusion",
    "extract",
    "trainer_base",
    "callbacks_base",
    # primitives — helpers
    "spec",
    "can_bus",
    "graph_dm",
    "fusion_dm",
    "curriculum",
    # primitives — class paths
    "GAT",
    "VGAE",
    "DGI",
    "BANDIT",
    "DQN",
    "MLP_FUSION",
    "MOE_FUSION",
    "WAVG_FUSION",
    "FOCAL",
    "CE",
    "WEIGHTED_CE",
    "VGAE_TASK",
    "CURRICULUM_LOSS",
    "LINEAR_RAMP",
    "SOFT_LABEL_DISTILLATION",
    "FEATURE_DISTILLATION",
    "SCORE_RANDOM",
    "SCORE_VGAE",
    "CAN_BUS",
    "GRAPH_DM",
    "FUSION_DM",
    "REWARD",
    "REWARD_MINIMAL",
]
